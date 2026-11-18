import numpy as np

class LatticeDict:
    def __init__(self, lattice, Nx, Ny, bd_condition):
        self.lattice = lattice
        self.Nsite = Nx*Ny
        self.Nx = Nx
        self.Ny = Ny
        self.bd_condition = bd_condition
        self.site_indices = {}
        self.positions = [] # position of sites
        
        if lattice == "Square":
            # The following are used for finding neighboring sites
            self.offset_nn = [(1,0), (0,1)]
            self.offset_nextnn = [(1,1), (1,-1)]
            self.offset_tnn = [(2,0), (0,2)]
            self.offset_Heisnn = [(1,0), (-1,0), (0,1), (0,-1)]
            self.avec1 = np.array([1,0])
            self.avec2 = np.array([0,1])
            self.bvec1 = np.array([2*np.pi, 0])
            self.bvec2 = np.array([0, 2*np.pi])
            self._init_Lattice_fill_sites(Nx, Ny) #update site information

            self.neighbor_pairs_Heisnn = self._get_neighbor_pairs(Nx,Ny,4, self.find_nneighbor_Heis)
        elif lattice == "Triangle": 
            # The following are used for finding neighboring sites
            self.offset_nn = [(1,0), (-1,1), (0,-1)]
            self.offset_nextnn = [(1,1), (-2,1), (1,-2)]
            self.offset_tnn = [(2,0), (-2,2), (0,-2)]
            self.offset_Heisnn = [(1,0), (-1,0), (-1,1), (1,-1), (0,-1), (0,1)]
            self.avec1 = np.array([1,0]) #b1 = np.array([2*np.pi, -2*np.pi/np.sqrt(3)])
            self.avec2 = np.array([1/2,np.sqrt(3)/2]) #b2 = np.array([0, 4*np.pi/np.sqrt(3)])
            self.bvec1 = np.array([2*np.pi, -2*np.pi/np.sqrt(3)])
            self.bvec2 = np.array([0, 4*np.pi/np.sqrt(3)])
            self._init_Lattice_fill_sites(Nx, Ny) #update site information

            self.neighbor_pairs_Heisnn = self._get_neighbor_pairs(Nx,Ny,6, self.find_nneighbor_Heis)

        else:
            raise Exception("The currently support lattice geometries are: lattce = [\"Square\",\"Triangle\"].")
            
    
    def _init_Lattice_fill_sites(self, Nx, Ny):
        index = 0
        for j in range(Ny):
            for i in range(Nx):
                self.site_indices[(i, j)] = index
                self.positions.append(i * self.avec1 + j * self.avec2)
                index += 1

    # avoid if..elif statement, this can potentially slower GPU calculation. find_nneighbor, find_nneighbor_Heis are used in every loops.
    def find_nneighbor(self, site, N_x, N_y):
        (i,j) = site
        nnset = []
        if self.bd_condition == "PBC":
            nnset = [((i + ioff)%N_x, (j + joff)%N_y) for ioff, joff in self.offset_nn]
        elif self.bd_condition == "OBC":
            nnset = [((i + ioff), (j + joff)) for ioff, joff in self.offset_nn]
        return nnset

    def find_nextnneighbor(self, site, N_x, N_y):
        (i,j) = site
        nnnset = []
        if self.bd_condition == "PBC":
            nnnset = [((i + ioff)%N_x, (j + joff)%N_y) for ioff, joff in self.offset_nextnn]
        elif self.bd_condition == "OBC":
            nnnset = [((i + ioff), (j + joff)) for ioff, joff in self.offset_nextnn]
        return nnnset

    def find_tnneighbor(self, site, N_x, N_y):
        (i,j) = site
        tnnset = []
        if self.bd_condition == "PBC":
            tnnset = [((i + ioff)%N_x, (j + joff)%N_y) for ioff, joff in self.offset_tnn]
        elif self.bd_condition == "OBC":
            tnnset = [((i + ioff), (j + joff)) for ioff, joff in self.offset_tnn]
        return tnnset

    # find Heisenberg neighbour
    def find_nneighbor_Heis(self, site, N_x, N_y):
        (i,j) = site
        Heisnnset = []
        if self.bd_condition == "PBC":
            Heisnnset = [((i + ioff)%N_x, (j + joff)%N_y) for ioff, joff in self.offset_Heisnn]
        elif self.bd_condition == "OBC":
            Heisnnset = [((i + ioff), (j + joff)) for ioff, joff in self.offset_Heisnn]
        return Heisnnset

    def _get_neighbor_pairs(self, N_x, N_y, num_bonds, find_neighbor_func):
        bond_lists = [[] for _ in range(num_bonds)]
    
        for j in range(N_y):
            for i in range(N_x):
                r1 = i + N_x * j
                Heisnn = find_neighbor_func((i, j), N_x, N_y)
                for bond, (i2, j2) in enumerate(Heisnn):
                    if 0 <= i2 <= N_x-1 and 0 <= j2 <= N_y-1:
                        r2 = i2 + N_x * j2
                        bond_lists[bond].append((r1, r2))
    
        bond_arrays = [np.array(pairs, dtype=np.int32) for pairs in bond_lists]
        return bond_arrays