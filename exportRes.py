#Klaus Roppert, IGTE, 2022
import numpy as np
import h5py


class EXPORT:
    def __init__(self, nodesPerEl, nElements, nNodes, dim, U, geom, connec_plot, n_dof_p_node): # "self" will be replaces by the name of the object (e.g. here "domain")
        ''' 
        Exports the results in HDF5 file  
        
        parameters
        ----------
        :dim: dimension of the domain
        :U: results vector
        :R: results matrix
        :geom: node coordinates matrix
        :connec_plot: node connectivity matrix (note that connec_plot and connec_geom are alike but not the same!)
        :domainName: domain name
        :nodesPerEl: number of nodes per element
        :nNodes: number of nodes in the domain
        :nElements: number of elements in the domain
        '''     
    
        self.nodesPerEl = nodesPerEl
        self.nElements = nElements
        self.nNodes = nNodes
        self.dim = dim
        self.U = U
        self.geom = geom
        self.connec_plot = connec_plot
        self.isScalarField = False
        self.n_dof_p_node = n_dof_p_node
        self.resultName = ''
        #print(( 'nNodes=', self.nNodes,'nElements=', self.nElements,'nodesPerEl', self.nodesPerEl))
        
        self.isScalarField = True if (self.n_dof_p_node == 1) else False
        if( self.isScalarField ):
            # SCALAR field
            self.R = np.zeros((self.nNodes,1))
            self.resultName = 'elecPotential'
            for i in range(self.nNodes):                      # (e.g. nodes = rows -> 0-14) 
                self.R[i,0]=self.U[i,]
        else:
            # VECTOR field
            self.R = np.zeros((self.nNodes,2))
            self.resultName = 'mechDisplacement'
            for i in range(self.nNodes):                      # (e.g. nodes = rows -> 0-14) 
                for j in range(self.dim):                     # (e.g. dimensions = columns -> 0-1)
                    self.R[i,j]=self.U[2*i+j,]
        

#        #DEBUG
#        print('\n--------------------------------------------------')
#        print('\nR=\n', self.R, 'results matrix\n')
#        print('\n--------------------------------------------------')
#        print('\nconnec_plot=\n', self.connec_plot, 'node connectivity matrix\n')  
        
        if nodesPerEl == 9:                                           # |swaps columns of the connectivity matrix (needed for the solver)
            for i in range(self.connec_plot.shape[0]):                # |
                self.a1 = self.connec_plot[i,2]                       # |                     6---7---8            3---6---2
                self.a2 = self.connec_plot[i,8]                       # |                     |       |            |       | 
                self.a3 = self.connec_plot[i,6]                       # |                     3   4   5     ->     7   8   5
                self.a4 = self.connec_plot[i,1]                       # |                     |       |            |       | 
                self.a6 = self.connec_plot[i,7]                       # |              pyFE2D 0---1---2         h5 0---4---1
                self.a7 = self.connec_plot[i,3]
                self.a8 = self.connec_plot[i,4]
                self.connec_plot[i,1] = self.a1  
                self.connec_plot[i,2] = self.a2
                self.connec_plot[i,3] = self.a3  
                self.connec_plot[i,4] = self.a4
                self.connec_plot[i,6] = self.a6  
                self.connec_plot[i,7] = self.a7
                self.connec_plot[i,8] = self.a8
        elif nodesPerEl == 4:                                         # |swaps the last two columns of the connectivity matrix (needed for the solver)
            for i in range(self.connec_plot.shape[0]):                # |                     2-------3            3-------2 
                self.a2 = self.connec_plot[i,3]                       # |                     |       |            |       | 
                self.a3 = self.connec_plot[i,2]                       # |                     |       |     ->     |       | 
                self.connec_plot[i,2] = self.a2                       # |                     |       |            |       | 
                self.connec_plot[i,3] = self.a3                       # |              pyFE2D 0-------1         h5 0-------1
        else:
            raise ValueError('ERROR! Expected only 4 or 9 node quadrilateral elements.')
        
        
        self.connec_plot = self.connec_plot+np.ones((self.nElements,nodesPerEl))   # adds +1 to every element in the matrix (changes indexing from 0,1,... to 1,2,...)
        
        self.geom_plot = np.zeros((self.nNodes,3))
        self.geom_plot[:,:-1] = self.geom
      
    def writeResults(self):
        '''
        Writes a HDF5 file containing mesh and results.       
              
        parameters
        ----------
        :dim: dimension of the domain
        :R: results matrix
        :geom_plot: node coordinates matrix
        :connec_plot: node connectivity matrix (note that connec_plot and connec_geom are alike but not the same!)
        :nodesPerEl: number of nodes per element
        :nElements: number of elements in the domain
        '''
        
        self.file = h5py.File('results.cfs', 'w')
       
#===================================================================================================
#====================================================================================== SET GROUPS
#===================================================================================================
        # DATABASE
        grpData                      = self.file.create_group("DataBase")
        grpDataMS                    = self.file.create_group("/DataBase/MultiSteps")
        grpDataMS1                   = self.file.create_group("/DataBase/MultiSteps/1")

        # MESH
        grpMesh                      = self.file.create_group("Mesh")
        grpElements                  = self.file.create_group("/Mesh/Elements")
        grpGroups                    = self.file.create_group("/Mesh/Groups")
        grpNodes                     = self.file.create_group("/Mesh/Nodes")
        grpRegions                   = self.file.create_group("/Mesh/Regions")
        grpDomain                    = self.file.create_group("/Mesh/Regions/Domain")   # %s % self.domain    
        
        # RESULTS
        grpResults                   = self.file.create_group("Results")
        grpResMesh                   = self.file.create_group("/Results/Mesh")
        grpResMultistep              = self.file.create_group("/Results/Mesh/MultiStep_1")
        grpResDescription            = self.file.create_group("/Results/Mesh/MultiStep_1/ResultDescription")
        grpRes       = self.file.create_group(f'/Results/Mesh/MultiStep_1/ResultDescription/{self.resultName}')
        grpResStep1                  = self.file.create_group("/Results/Mesh/MultiStep_1/Step_1")
        grpRes_Step1 = self.file.create_group(f'/Results/Mesh/MultiStep_1/Step_1/{self.resultName}')
        grpResDomain_Step1           = self.file.create_group(f'/Results/Mesh/MultiStep_1/Step_1/{self.resultName}/Domain')
        grpResNodes_Step1            = self.file.create_group(f'/Results/Mesh/MultiStep_1/Step_1/{self.resultName}/Domain/Nodes')
        grpResStep2                  = self.file.create_group("/Results/Mesh/MultiStep_1/Step_2")
        grpRes_Step2 = self.file.create_group(f'/Results/Mesh/MultiStep_1/Step_2/{self.resultName}')
        grpResDomain_Step2           = self.file.create_group(f'/Results/Mesh/MultiStep_1/Step_2/{self.resultName}/Domain')
        grpResNodes_Step2            = self.file.create_group(f'/Results/Mesh/MultiStep_1/Step_2/{self.resultName}/Domain/Nodes')
        
        
#=======================================================================================================
#====================================================================================== SET ATTRIBUTES
#=======================================================================================================
        # MESH
        grpMesh.attrs["Dimension"] = np.uint32(self.dim) # 2
        
        # MESH/ELEMENTS
        if self.nodesPerEl == 4:
            grpElements.attrs["Num_QUAD4"]    = np.uint32(self.nElements)                # 4 node quadrangle
        elif self.nodesPerEl == 9:
            grpElements.attrs["Num_QUAD9"]    = np.uint32(self.nElements)                # 9 node quadrangle
        else:
            raise ValueError('ERROR! Expected only 4 or 9 node quadrangle elements.') 
        
        grpElements.attrs["Num1DElems"]     = np.uint32(0)
        grpElements.attrs["Num2DElems"]     = np.uint32(self.nElements)
        grpElements.attrs["Num3DElems"]     = np.uint32(0)
        grpElements.attrs["NumElems"]       = np.uint32(self.nElements)
        grpElements.attrs["Num_HEXA20"]     = np.uint32(0)
        grpElements.attrs["Num_HEXA27"]     = np.uint32(0)
        grpElements.attrs["Num_HEXA8"]      = np.uint32(0)
        grpElements.attrs["Num_LINE2"]      = np.uint32(0)
        grpElements.attrs["Num_LINE3"]      = np.uint32(0)
        grpElements.attrs["Num_POINT"]      = np.uint32(0)
        grpElements.attrs["Num_POLYGON"]    = np.uint32(0)
        grpElements.attrs["Num_POLYHEDRON"] = np.uint32(0)
        grpElements.attrs["Num_PYRA13"]     = np.uint32(0)
        grpElements.attrs["Num_PYRA14"]     = np.uint32(0)
        grpElements.attrs["Num_PYRA5"]      = np.uint32(0)
#        grpElements.attrs["Num_QUAD4"]      = np.uint32(0)
        grpElements.attrs["Num_QUAD8"]      = np.uint32(0)
#        grpElements.attrs["Num_QUAD9"]      = np.uint32(0)
        grpElements.attrs["Num_TET10"]      = np.uint32(0)
        grpElements.attrs["Num_TET4"]       = np.uint32(0)
        grpElements.attrs["Num_TRIA3"]      = np.uint32(0)
        grpElements.attrs["Num_TRIA6"]      = np.uint32(0)
        grpElements.attrs["Num_UNDEF"]      = np.uint32(0)
        grpElements.attrs["Num_WEDGE15"]    = np.uint32(0)
        grpElements.attrs["Num_WEDGE18"]    = np.uint32(0)
        grpElements.attrs["Num_WEDGE6"]     = np.uint32(0)
        grpElements.attrs["QuadraticElems"] = np.int32(0)
      
        # MESH/NODES
        grpNodes.attrs["NumNodes"] = np.uint32(self.nNodes)
        
        # MESH/REGIONS/DOMAIN
        grpDomain.attrs["Dimension"] = np.uint32(self.dim)  # e.g. here =2
        
        # /RESULTS/MESH
        grpResMesh.attrs["ExternalFiles"] = np.int32(0)
        
        # /RESULTS/MESH/MULTISTEP_1 
        grpResMultistep.attrs["AnalysisType"]  = bytes("transient", 'ascii') # r'transient'
        grpResMultistep.attrs["LastStepNum"]   = np.uint32(2)
        grpResMultistep.attrs["LastStepValue"] = 1.0

        # /RESULTS/MESH/MULTISTEP_1/STEP_1
        grpResStep1.attrs["StepValue"] = 0.1
        
        # /RESULTS/MESH/MULTISTEP_1/STEP_2
        grpResStep2.attrs["StepValue"] = 1.0
      
        # /DATABASE/MULTISTEP/1
        grpDataMS1.attrs["AnalysisType"]  = bytes("transient", 'ascii')
        grpDataMS1.attrs["AccTime"]  = 1.0
        grpDataMS1.attrs["Completed"]  = np.int32(1)
        
#=====================================================================================================
#====================================================================================== SET DATASETS 
#=====================================================================================================
        # MESH/ELEMENTS
        dsetConnectivity = grpElements.create_dataset("Connectivity", (self.nElements,self.nodesPerEl), dtype='i')     # creates empty connectivity matrix (nElements, nodesPerEl)
        dsetConnectivity[...] = self.connec_plot
        dsetTypes = grpElements.create_dataset("Types", (self.nElements,), dtype='i')                                  # creates empty el. types matrix (nElements, ) 
        if self.nodesPerEl == 4:                                                                                       
            dsetTypes[...] = 6                                                                                         # sets element type for paraview plug in (6 = QUAD4)
        elif self.nodesPerEl == 9:
            dsetTypes[...] = 8                                                                                         # sets element type for paraview plug in (8 = QUAD9)
        else:
            raise ValueError('ERROR! Expected only 4 or 9 node quadrangle elements.') 
        
        
        # MESH/NODES
        dsetCoordinates      = grpNodes.create_dataset("Coordinates", (self.nNodes,3))                      # creates node coordinates matrix (nNodes, dimension)
        dsetCoordinates[...] = self.geom_plot
        
        # MESH/REGIONS/DOMAIN
        dsetElements      = grpDomain.create_dataset("Elements", (self.nElements,), dtype='i')              # (nElements, )
        dsetElements[...] = np.arange(1,self.nElements+1,1)                                                 # renumbers the elements (e.g. from 0...7 to 1...8)
        dsetNodes         = grpDomain.create_dataset("Nodes", (self.nNodes,), dtype='i')                    # (nNodes, )
        dsetNodes[...]    = np.arange(1,self.nNodes+1,1)                                                    # renumbers the nodes (e.g. from 0...14 to 1...15)
        
        # /RESULTS/MESH/MULTISTEP_1/RESULTDESCRIPTION/MECHDISPLACEMENT
        dt                    = h5py.special_dtype(vlen=str)  #h5py.string_dtype(encoding='utf-8')                                              # sets variableType = string with variable lenght
        if(self.isScalarField):
            dsetDOFNames      = grpRes.create_dataset('DOFNames',    (1,), dtype=dt )
            dsetDOFNames[...] = ''
        else:
            dsetDOFNames      = grpRes.create_dataset('DOFNames',    (2,), dtype=dt )
            dsetDOFNames[...] = 'x', 'y'
        dsetDefinedOn         = grpRes.create_dataset("DefinedOn",   (1,), dtype='uint32')
        dsetDefinedOn[...]    = 1
        dsetEntityNames       = grpRes.create_dataset('EntityNames', (1,), dtype=dt )
        dsetEntityNames[...]  = 'Domain'
        dsetEntryType         = grpRes.create_dataset("EntryType",   (1,), dtype='uint32')
        
        if(self.isScalarField):
            dsetEntryType[...]= 1
        else:
            dsetEntryType[...]= 2
        dsetNumDOFs           = grpRes.create_dataset("NumDOFs",     (1,), dtype='uint32')
        if(self.isScalarField):
            dsetNumDOFs[...]  = 1
        else:
            dsetNumDOFs[...]  = 2
        dsetStepNumbers       = grpRes.create_dataset("StepNumbers", (2,), dtype='uint32')
        dsetStepNumbers[...]  = np.arange(1,3,1)
        dsetStepValues        = grpRes.create_dataset("StepValues",  (2,), dtype='f')
        dsetStepValues[...]   = [0.1,1.0]
        dsetUnit              = grpRes.create_dataset('Unit',        (1,), dtype=dt )
        dsetUnit[...]         = 'm'
      
        
        # /RESULTS/MESH/MULTISTEP_1/STEP_1/MECHDISPLACEMENT/DOMAIN/NODES
#        dsetImag_Step1 = grpResNodes_Step1.create_dataset("Imag", (self.nNodes,3))#, dtype='f') # (15,3)
        if(self.isScalarField):
            dsetReal_Step1 = grpResNodes_Step1.create_dataset("Real", (self.nNodes,1))
        else:
            dsetReal_Step1 = grpResNodes_Step1.create_dataset("Real", (self.nNodes,2))
      
        
        # /RESULTS/MESH/MULTISTEP_1/STEP_2/MECHDISPLACEMENT/DOMAIN/NODES
#        dsetImag_Step2      = grpResNodes_Step2.create_dataset("Imag", (self.nNodes,3)) #, dtype='f') # (15,3)
        if(self.isScalarField):
            dsetReal_Step2 = grpResNodes_Step2.create_dataset("Real", (self.nNodes,1))
        else:
            dsetReal_Step2 = grpResNodes_Step2.create_dataset("Real", (self.nNodes,2))
        dsetReal_Step2[...] = self.R

#=====================================================================================================
#====================================================================================== CLOSE FILE 
#=====================================================================================================
        self.file.close()
        return ('Results have been written!')
