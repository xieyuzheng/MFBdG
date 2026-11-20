"""Momentum-block operations for the original onsite and covariance solvers.

MFSolver owns the shared iteration, backtracking and recovery policy;
bond_bdg/barrier_descent supply constraint calculations and predictors.
This adapter supplies Fourier contractions and momentum-block operations.
"""
import numpy as np
from scipy.linalg import cho_factor, eigh
from scipy.special import expit, entr

from CPU import bond_bdg as constraints
from CPU.solver.solver1d import secant_method


class MomentumBackend:
    def __init__(self, model, runtime=None, lane=0, batch_size=128, pairing_sector='mixed'):
        from CPU import cell_bdg
        self.cell, self.model = cell_bdg, model
        self.nsite, self.nk, self.m = model.nsite, model.nk, model.nmode
        self.runtime, self.lane = runtime, lane
        self.pairing_sector = pairing_sector
        self.q = np.concatenate(list(model.momenta(batch_size)))
        ix=np.arange(model.nk)
        self.negative = (-ix%model.nkx) + model.nkx*((-(ix//model.nkx))%model.nky)
        self.phase = np.exp(1j*self.q @ model.shift.T)
        self.up=2*np.arange(self.nsite)
        zeros=(np.zeros(self.m),np.zeros(len(model.a)),np.zeros(len(model.a)))
        self.h0=cell_bdg.bloch_blocks(model,zeros,self.q)[2]
        self.bonds=~model.onsite
        self.indices=(np.flatnonzero(self.bonds),np.arange(len(model.a)))
        self.bond_weights=model.coupling[self.bonds]

    def _map_k(self, function):
        """Consume each k-point inside its worker; never return eigenvectors."""
        def work(bounds):
            first,last=bounds
            return [function(index) for index in range(first,last)]
        workers=1 if self.runtime is None else self.runtime.plan.momentum_workers
        bounds=self.cell._momentum_partitions(self.model,workers)
        groups=[work(bounds[0])] if workers==1 else self.runtime.map_momentum(work,bounds,self.lane)
        return [value for group in groups for value in group]

    def _modes(self,h,gap,T,mu,index):
        bdg=np.zeros((2*self.m,2*self.m),complex)
        bdg[:self.m,:self.m]=h[index]-mu*np.eye(self.m)
        bdg[:self.m,self.m:]=gap[index]
        bdg[self.m:,:self.m]=gap[index].conj().T
        bdg[self.m:,self.m:]=-h[self.negative[index]].T+mu*np.eye(self.m)
        try:
            energies,vectors=np.linalg.eigh(bdg)
        except np.linalg.LinAlgError:
            energies,vectors=eigh(bdg,driver='ev')
        return energies,vectors,expit(-energies/T)

    def contractions(self,h,gap,T,mu=0.):
        energies=np.empty((self.nk,2*self.m))
        rho=np.empty((self.nk,self.m,self.m),complex)
        pair=np.empty_like(rho)
        def evaluate(index):
            e,v,f=self._modes(h,gap,T,mu,index)
            G=(v.conj()*f) @ v.T
            del v
            energies[index]=e
            rho[index]=G[:self.m,:self.m]
            pair[index]=G[self.m:,:self.m].T
        self._map_k(evaluate)
        return energies,rho,(pair-pair[self.negative].swapaxes(-1,-2))/2

    def state(self, rho, pair, energies, T, mu):
        n, chi, anomalous = self.order(rho, pair)
        local_r, local_p = self.local_matrices(rho, pair)
        sites = np.arange(self.nsite)
        local_r = local_r.reshape(self.nsite, 2, self.nsite, 2)[sites, :, sites, :]
        local_p = local_p.reshape(self.nsite, 2, self.nsite, 2)[sites, :, sites, :]
        f = expit(-energies/T)
        entropy = np.sum(entr(f)+entr(1-f))/(2*self.nsite*self.nk)
        energy = self.energy(rho, pair)
        return dict(occupations=n, chi=chi, pair=anomalous,
                    rho_local=local_r, pair_local=local_p, density=self.density(rho),
                    filling=self.filling(rho), mu=mu, energy=energy, entropy=entropy,
                    free=energy-T*entropy, energies=energies)

    def project_pair(self,pair):
        pair=(pair-pair[self.negative].swapaxes(-1,-2))/2
        if self.pairing_sector=='singlet':
            out=np.zeros_like(pair)
            singlet=(pair[:,::2,1::2]-pair[:,1::2,::2])/2
            out[:,::2,1::2]=singlet
            out[:,1::2,::2]=-singlet
            pair=out
        return pair

    def local_matrices(self,rho,pair):
        # Only local blocks enter the original constraint functions.
        return rho.mean(axis=0),pair.mean(axis=0)

    def local_values(self,rho,pair,filling):
        return constraints._local_values(*self.local_matrices(rho,pair),filling)

    def density(self,rho):
        return rho.diagonal(axis1=-2,axis2=-1).real.mean(axis=0).reshape(-1,2).sum(axis=1)

    def filling(self,rho):
        return self.density(rho).mean()

    def initial(self,rho,pair,target):
        rho,pair=rho.copy(),pair.copy()
        rho[:,np.arange(self.m),np.arange(self.m)]+=(target-self.filling(rho))/2
        return rho,self.project_pair(pair)

    def add_fields(self,h,gap,fields):
        q,bx,by,bz=fields.T
        up,dn=self.up,self.up+1
        h[:,up,up]+=q+bz; h[:,dn,dn]+=q-bz
        h[:,up,dn]+=bx-1j*by; h[:,dn,up]+=bx+1j*by

    def add_barrier(self,h,gap,values,cap,tau,interior=False):
        normal=np.zeros((self.m,self.m),complex); pairing=np.zeros_like(normal)
        fields=constraints.add_local_barrier(normal,pairing,values,cap,tau,interior=interior)
        self.add_fields(h,gap,fields)
        return fields

    def barrier(self,rho,pair,target,cap,tau):
        return constraints._local_barrier_energy(*self.local_matrices(rho,pair),target,cap,tau)

    def lift(self,order):
        n,chi,pair=order; model=self.model
        rho=np.broadcast_to(np.diag(n),(self.nk,self.m,self.m)).astype(complex).copy()
        P=np.zeros_like(rho)
        self.cell._add_links(rho,model.a,model.b,chi*self.phase.conj())
        self.cell._add_links(rho,model.b,model.a,chi.conj()*self.phase)
        self.cell._add_links(P,model.a,model.b,pair*self.phase)
        self.cell._add_links(P,model.b,model.a,-pair*self.phase.conj())
        return rho,self.project_pair(P)

    def order(self,rho,pair):
        model=self.model
        n=rho.diagonal(axis1=-2,axis2=-1).real.mean(axis=0)
        chi=(self.phase*rho[:,model.a,model.b]).mean(axis=0)
        pair=(self.phase.conj()*pair[:,model.a,model.b]).mean(axis=0)
        return n,chi,pair

    def build(self,rho,pair):
        return self.cell.bloch_blocks(self.model,self.order(rho,pair),self.q)[:2]

    def vector(self,rho,pair):
        _,chi,pair=self.order(rho,pair)
        values=np.r_[chi[self.bonds],pair]
        return np.r_[values.real,values.imag]

    def proposal(self,rho,pair,chi,anomalous):
        n,c,p=self.order(rho,pair)
        c[self.bonds],p[:]=chi,anomalous
        return self.cell.bloch_blocks(self.model,(n,c,p),self.q)[:2]

    def perturb(self,h,gap,noise):
        model=self.model; selected=self.bonds
        a,b=model.a[selected],model.b[selected]; phase=self.phase[:,selected]
        self.cell._add_links(h,a,b,noise[0]*phase)
        self.cell._add_links(h,b,a,noise[0].conj()*phase.conj())
        self.cell._add_links(gap,a,b,noise[1]*phase)
        self.cell._add_links(gap,b,a,-noise[1]*phase.conj())

    def occupations(self,rho,pair):
        def evaluate(index):
            G=np.zeros((2*self.m,2*self.m),complex)
            G[:self.m,:self.m]=rho[index]
            G[self.m:,:self.m]=pair[index].T
            G[:self.m,self.m:]=pair[index].conj()
            G[self.m:,self.m:]=np.eye(self.m)-rho[self.negative[index]].T
            return np.linalg.eigvalsh(G)
        return np.asarray(self._map_k(evaluate))

    def energy(self,rho,pair):
        n,chi,P=self.order(rho,pair); model=self.model
        kinetic=np.sum(self.h0*rho).real/self.nk
        interaction=model.coupling @ (n[model.a]*n[model.b]-abs(chi)**2+abs(P)**2)
        return (kinetic+interaction)/self.nsite

    def repair(self,rho,pair,local,target):
        current=self.local_values(rho,pair,target)
        change=local-current
        norm=np.max(abs(change[:,0])+np.linalg.norm(change[:,1:4],axis=1))/2
        margin=min(target/2,1-target/2); weight=margin/(margin+norm)
        rho=weight*rho+(1-weight)*np.eye(self.m)*target/2
        pair=weight*pair
        dn,mx,my,mz=(weight*local-self.local_values(rho,pair,target)).T
        up,dw=self.up,self.up+1
        rho[:,up,up]+=(dn+mz)/2;rho[:,dw,dw]+=(dn-mz)/2
        rho[:,up,dw]+=(mx+1j*my)/2;rho[:,dw,up]+=(mx-1j*my)/2
        return rho,pair

    def residual(self,dr,dk):
        shape=(self.model.nky,self.model.nkx,self.m,self.m)
        r=np.fft.ifft2(dr.reshape(shape),axes=(0,1)).reshape(dr.shape)
        p=(np.fft.fft2(dk.reshape(shape),axes=(0,1))/self.nk).reshape(dk.shape)
        local=max(abs(r[0].reshape(self.nsite,2,self.nsite,2)[np.arange(self.nsite),:,np.arange(self.nsite),:]).max(),
                  abs(p[0].reshape(self.nsite,2,self.nsite,2)[np.arange(self.nsite),:,np.arange(self.nsite),:]).max())
        for value in (r,p):
            for s in (0,1):
                for t in (0,1):
                    value[0,self.up+s,self.up+t]=0
        return np.array([local,abs(r).max(),abs(p).max()])

    def solve_mu(self,h,gap,T,target,mu,tol,normal_first=False):
        errors={}
        def residual(value):
            if value not in errors:
                def number(index):
                    _,v,f=self._modes(h,gap,T,value,index)
                    # Keep only orbital particle counts after summing eigenmodes.
                    return np.sum(abs(v[:self.m])**2*f,axis=1)
                errors[value]=np.sum(self._map_k(number))/(self.nsite*self.nk)-target
            return errors[value]
        for normal in (normal_first,not normal_first):
            seed=mu
            if normal:
                levels=np.asarray(self._map_k(lambda index:np.linalg.eigvalsh(h[index])))
                seed=secant_method(lambda x:2*expit((x-levels)/T).mean()-target,x1=mu,ftol=tol,print_info=False)
            value=secant_method(residual,x1=seed,ftol=tol,print_info=False)
            if np.isfinite(value) and abs(residual(value))<=tol:
                return value
        raise RuntimeError('Momentum number equation did not converge.')

    def field_response(self,h,gap,T,mu):
        if self.nk==1:
            return constraints._field_response(h[0],gap[0],T,mu)
        # Exact divided-difference derivative of the Fermi matrix. Limit the
        # temporary source matrices; the original predictor can fall back to
        # its safeguarded residual direction if a response would be too large.
        count=4*self.nsite; dim=2*self.m
        if count*dim*dim*16>256*1024**2:
            raise MemoryError('Use the safeguarded field predictor for this cell size.')
        sources=np.zeros((count,dim,dim),complex)
        for site in range(self.nsite):
            for channel in range(4):
                fields=np.zeros((self.nsite,4));fields[site,channel]=1
                normal=np.zeros((self.m,self.m),complex);pairing=np.zeros_like(normal)
                constraints._add_onsite_fields(normal,pairing,fields)
                sources[4*site+channel]=np.block([[normal,pairing],[pairing.conj().T,-normal.T]])
        def evaluate(index):
            e,v,occ=self._modes(h,gap,T,mu,index)
            de=e[:,None]-e[None,:]
            near=abs(de)<1e-10*max(1.,abs(e).max())
            divided=np.divide(occ[:,None]-occ[None,:],de,out=np.zeros_like(de),where=~near)
            mean=(occ[:,None]+occ[None,:])/2
            divided[near]=(-mean*(1-mean)/T)[near]
            transformed=v.conj().T @ sources @ v
            del v
            return .5*np.einsum('aij,bji,ij->ab',transformed,transformed,divided,optimize=True).real/self.nk
        response=np.zeros((count,count))
        # Source transforms dominate memory; keep their original serial traversal.
        for index in range(self.nk):
            response-=evaluate(index)
        response=(response+response.T)/2
        response.flat[::count+1]+=1e-9
        return cho_factor(response,check_finite=False)
