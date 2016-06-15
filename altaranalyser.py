## Script made to load ALTAR result and plot it with the csi multisolveur

# Import externals
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import pickle
import glob
import sys
import h5py

# Import personal libraries
import csi.TriangularTents as triangular_fault
#import csi.TriangularPatches as triangular_fault 
import csi.fault3D as rectFault
import csi.geodeticplot as geoplt
import csi.gps as gr
import csi.insar as ir
import csi.opticorr as cr
import csi.multifaultsolve as multiflt
import csi.faultpostproc as faultpp



class altaranalyser(object):

    def __init__(self, AltarFile, DataFile, SlvFile):

        '''
        Class initialization routine.

        Args:
            * AltarFile         : .h5 file containing ALTAR solution.
            * DataFile          : pickle file containing datasets object
            * SlvFile           : pickle file containing multisolveur object 


        '''

        print('-------------------------------------')
        print('----  Initializing Altar Analyser ---')
        print('-------------------------------------')

        for file in [AltarFile, DataFile, SlvFile]:
            if not os.path.isfile(file):
                print('\n ERROR, ERROR')
                print("{} file not found !".format(file))
                sys.exit(0)


        self.slvfile = SlvFile
        self.altarfile = AltarFile
        self.datafile = DataFile

        self.workdir = AltarFile.rpartition('/')[0] + '/'
        self.NewM = False
        self.NewSynth = False
        

        # Some parameters
        bounds = {}
        bounds['strikeslip'] = (-10., 10.0)
        bounds['dipslip'] = (-5.0, 5.0)
        bounds['insar'] = (-1.0, 1.0)
        bounds['opticorr'] = (-5.0, 5.0)
        self.bounds = bounds
        # All done
        return 

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

    def LoadSlvFile(self):

        '''
        Load slv file in self.slv
        '''

        print('-------------------------------------')
        print('------  Loading SLV pickle file  ----')
        print('-------------------------------------')


        
        slvfile = self.slvfile

        with open(slvfile,'r') as iput:
            slv = pickle.load(iput)

        self.slv = slv

        # All done
        return

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

    def LoadDataFile(self):

        '''
        Load datasets file in self.datasets
        '''

        print('-------------------------------------')
        print('-----  Loading Data pickle file  ----')
        print('-------------------------------------')

        datafile = self.datafile

        with open(datafile,'r') as iput:
            datasets = pickle.load(iput)

        self.datasets = datasets

        # All done
        return


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

    def LoadALTARresult(self, Complete = False):

        '''
        Load ALTAR h5py result file
        
        OPT             : Complete = True if you want to load every variable in the h5py file
                          It can result in an extremely high memory cost 
        '''

        print('-------------------------------------')
        print('------   Loading ALTAR file    ------')
        print('-------------------------------------')

        altfile = self.altarfile
         
        f = h5py.File(altfile,'r')

        if Complete:
            print('------ WARNING WARNING WARNING ------')
            print('----- You chose Complete = True -----')

            self.AltarSamples = f['Sample Set'].value # Samples
            self.AltarMmean = self.AltarSamples.mean(0) 
            self.AltarDataLogLLK = f['Data Log-likelihood'].value
            self.AltarCovariance = f['Covariance'].value
            self.AltarPostLogLLK = f['Posterior Log-likelihood'].value
            self.AltarMmedian = np.median(self.AltarSamples,0)
            self.AltarMmax = np.zeros((len(self.AltarMmean)))
            for i in range(len(self.AltarMmean)):
                h, bin_edges = np.histogram(self.AltarSamples[:,i],bins=100)
                ind = bin_edges
                db = (bin_edges[1] - bin_edges[0])/2.
                self.AltarMmax[i] = bin_edges[h.argmax()] + db

        else:
            self.AltarMmean = f['Sample Set'].value.mean(0)
        f.close()


        # All done
        return 


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def ReOrganizeMbyFault(self, Complete = False):

        '''
        If a result is organised by slipmode and if there are several faults, this method
        reorganize the result matrix by fault segment to be consistent with multisolveur:

        [fault1_SS, fault2_SS, fault1_DS, fault2_DS] -> [fault1_SS, fault1_DS, fault2_SS, fault2_DS]

        OPT         : Complete = True if a complete loading of the ALTAR h5py was done

        '''

        if not hasattr(self,'AltarMmean'):
                print('\n ERROR, ERROR')
                print('You must load ALTAR results first')
                sys.exit(0)

        info = self.slv.paramDescription

        N = 0
        
        slipmode = ['Strike Slip', 'Dip Slip', 'Tensile Slip', 'Coupling', 'Extra Parameters']

        # If all ALTAR variables are loaded 
        if Complete:
            Mtemp = np.zeros((self.AltarSamples.shape)) 
            Mtemp2 = np.zeros((self.AltarMmax.shape))       
                  
            for mode in slipmode:
                for fault in self.slv.faults:
                    if info[fault.name][mode].replace(' ','') != 'None':
                        ib = int(info[fault.name][mode].replace(' ','').partition('-')[0])
                        ie = int(info[fault.name][mode].replace(' ','').partition('-')[2])
                        Mtemp[:,ib:ie] = self.AltarSamples[:,N:N+ie-ib]
                        Mtemp2[ib:ie] = self.AltarMmax[N:N+ie-ib]
                        N += ie-ib
                  
            self.SavedSamples = self.AltarSamples.copy()
            self.AltarSamples = Mtemp
            self.AltarMmean = self.AltarSamples.mean(0) 
            self.AltarMmedian = np.median(self.AltarSamples,0)
            self.AltarMmax = Mtemp2

        else:
            Mtemp = np.zeros((self.AltarMmean.shape))            
            for mode in slipmode:
                for fault in self.slv.faults:
                    if info[fault.name][mode].replace(' ','') != 'None':
                        ib = int(info[fault.name][mode].replace(' ','').partition('-')[0])
                        ie = int(info[fault.name][mode].replace(' ','').partition('-')[2])
                        Mtemp[ib:ie] = self.AltarMmean[N:N+ie-ib]
                        N += ie-ib

            self.AltarMmean = Mtemp


        # All done
        return

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def ComputeSlipNorm(self,Reorganize = False):

        '''
        Compute the norm of the slip in each tent/patch.
        If the Altar results are organized by slip instead of by fault,
        and you haven't reorganize it yet, use Reorganize = True.

        Otherwise, you need to set ALTAR results in SLV object before.

        '''


        print('-------------------------------------')
        print('------- Compute Norm of Slip  -------')
        print('-------------------------------------')


        # Check something
        if not hasattr(self,'AltarMmean'):
            print('\n ERROR, ERROR')
            print('You need to load ALTAR result first!')
            sys.exit(0)

        Nparam = self.slv.nSlip/2
        strikeslip = np.zeros((Nparam))
        dipslip = np.zeros((Nparam))
        
        if Reorganize:
            strikeslip = AltarMmean[0:Nparam]
            dipslip = AltarMmean[Nparam:]
            self.slipNorm = np.sqrt(strikeslip**2+dipslip**2)

        else:
            if not self.NewM:
                print('\n ERROR, ERROR')
                print('You need to load ALTAR result in Slv object first !')
                sys.exit(0)

            k = 0
            tmpnorm = np.zeros((Nparam))
            for fault in self.slv.faults:
                tmp = np.linalg.norm(fault.slip,axis=1)
                tmpnorm[k:k+len(tmp)] = tmp
                k += len(tmp)
            self.slipNorm = tmpnorm
        
        # All done
        return 

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def SetSamplesInSLV(self, Parameter = 'mean'):

        print('-------------------------------------')
        print('--- Setting samples in slv object ---')
        print('-------------------------------------')

        if Parameter == 'mean':
            m_alt = self.AltarMmean
        elif Parameter == 'max':
            m_alt = self.AltarMmax
        elif Parameter == 'median':
            m_alt = self.AltarMmedian

        if m_alt.shape != self.slv.mpost.shape:
            print('\n ERROR, ERROR')
            print('slv.mpost and ALTAR results do not have the same number of parameters')
            sys.exit(0)

        self.slv.SetSolutionFromExternal(m_alt)
        self.slv.distributem()
        self.NewM = True

        # All done
        return 

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def LoadAll(self, ReOrganize = False, Complete = False, Parameter = 'mean', MwHist = False, ComputeCm = True):

        '''
        Load Pickle slv file, pickle data file, Altar result and set it in self.slv.mpost
        OPT         : * ReOrganize = True if you want to rearrange the ALTAR result by fault
                        instad of slipmode to be consistant with the SLV file

                      * Complete = True if you want to load every variable in the ALTAR h5py file
                        Be careful, it can result in a looooooong computation

        '''
        
        self.LoadSlvFile()
        self.LoadDataFile()
        self.LoadALTARresult(Complete = Complete)

        if Complete and ComputeCm:
            self.ComputeCm(ReOrganize = False)

        if ReOrganize:
            self.ReOrganizeMbyFault(Complete=Complete)

        self.SetSamplesInSLV(Parameter = Parameter)
        self.ComputeSlipNorm()
        #self.ComputeMw(ReOrganize = ReOrganize, Hist = MwHist)

        # All done
        return

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def BuildNewSynth(self,polycos = 'include', polysar = 'yes' ,vertical=True):

        ''' 
        Build synthetics with ALTAR result
        poly : To include CosiCorr poly, use 'include', to include SAR poly, use anything but None
        vertical : vertical for GPS
        '''

        print('-------------------------------------')
        print('------ Compute new synthetics  ------')
        print('-------------------------------------')


        # Check if ready
        if not self.NewM:
            print('\n ERROR, ERROR')
            print('You need to load ALTAR result in Slv object first !')
            sys.exit(0)

        if not hasattr(self,'datasets'):
            print('\n ERROR, ERROR')
            print('You need to load DataFile first !')
            sys.exit(0)

        faults = self.slv.faults

        # Moche, a changer, gere pb de polynomes si plusieurs failles
        self.adjustPolyPb()
        print 'prout'
        
        # Clean existing synthetics
        #for data in self.datasets:
        #    if hasattr(data,'synth'):
        #        if data.synth != None or data.synth != 'None' or data.synth.dtype != int:
        #            data.synth *= 0

        #if (self.datasets[0].vel_enu[:,2]==0).all():
        #    vertical = False
        #else:
        #    vertical = True
            

        # Build new synth       
        for data in self.datasets:
            print("Building {} synthetics".format(data.name))
            if data.name=='trilateration':
                data.buildsynth(self.slv.faults, poly=False,vertical=False)
            elif data.dtype == 'insar':
                data.buildsynth(self.slv.faults, poly=polysar,vertical=True)
            elif data.dtype == 'opticorr':
                data.buildsynth(self.slv.faults, poly=polycos,vertical=False)
            elif data.name == 'hudnut':
                data.buildsynth(self.slv.faults, poly=False,vertical=vertical) 

        self.NewSynth = True

        # All done
        return


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def plotNewSynth(self,figbegin=100, show = True):

        '''
        Plot InSAR, GPS and Trilateration synthethics along with data and residuals
        '''

        print('-------------------------------------')
        print('----- Plotting new synthetics -------')
        print('-------------------------------------')

        # Check if ready
        if not self.NewSynth:
            print('\n ERROR, ERROR')
            print('You need to compute new synthetics first!')
            sys.exit(0)

        datasets = self.datasets
        sardata = []
        gpsdata = []
        tridata = []

        plt.ion()

        for data in datasets:
            if data.dtype == 'insar':
                sardata.append(data)
            elif data.name == 'trilateration':
                tridata.append(data)
                continue
            elif data.dtype == 'gps':
                gpsdata.append(data)

        for sar in sardata:
            sarnorm = (np.abs(sar.vel)).max()

            sar.plot(figure=figbegin,data='data', decim=True, norm=(-sarnorm,sarnorm),faults=self.slv.faults)
            figbegin += 2
            sar.plot(figure=figbegin,data='synth', decim=True, norm=(-sarnorm,sarnorm),faults=self.slv.faults)
            figbegin += 2
            sar.plot(figure=figbegin,data='res', decim=True, norm=(-sarnorm*0.1,sarnorm*0.1),faults=self.slv.faults)
            figbegin += 2

        for gps in gpsdata:
            gps.plot(figure=figbegin,data=['data','synth'],color=['k','r'],legendscale=0.5,faults=self.slv.faults, scale=1.)
            figbegin += 2            
            #gps.plot(figure=figbegin,data='synth')
            #figbegin += 2            
            #gps.plot(figure=figbegin,data='res')
            #figbegin += 2            


        for tri in tridata:
            tri.plot(figure=figbegin,data=['data','synth'],color=['k','r'],legendscale=0.5,faults=self.slv.faults, scale = 1.)
            figbegin += 2            
            #tri.plot(figure=figbegin,data='synth')
            #figbegin += 2            
            #tri.plot(figure=figbegin,data='res')

        # All done
        return

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def plotNewModel(self,plotList = ['strikeslip','dipslip']):

        '''
        Plot slip model 
        Arg : 
            * plotList      : slipmodes to plot
        ''' 

        print('-------------------------------------')
        print('-------- Plotting new model ---------')
        print('-------------------------------------')

        # Check if ready
        if not self.NewM:
            print('\n ERROR, ERROR')
            print('You need to load ALTAR result in Slv object first!')
            sys.exit(0)
      
        if plotList.__class__ == str:
            plotList = [plotList]
       
        if  self.slv.faults[0].patchType == 'triangletent':
            fignum = 1
            for slipMode in plotList:
                gp = geoplt(fignum)
                colorbar = True
                for fault in self.slv.faults:
                    gp.faulttrace(fault, color='r')
                    gp.faultTents(fault, Norm=self.bounds[slipMode], colorbar=colorbar, slip=slipMode, method = 'scatter')
                    colorbar = False
                fignum += 1
                colorbar = True

        else: #if self.slv.faults[0].patchType == 'triangle':
            figNum = 1
            for slipMode in plotList:
                gp = geoplt(figure=figNum)
                colorbar = True
                for fault in self.slv.faults:
                    gp.faulttrace(fault, color='r')
                    gp.faultpatches(fault, Norm=self.bounds[slipMode], colorbar=colorbar, slip=slipMode)
                    colorbar = False
                figNum +=1
                colorbar = True
            
        plt.show()
        plt.ion()
        

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def plotMarginals(self,FigRange = 'All',xlim=(-1,40)):

        '''
        Plot marginals of Altar result
        Arg :
            * FigRange      : Range of parameters to plot (ex : [233,244])
        '''

        # Check something
        if not hasattr(self,'AltarSamples'):
            print('\n ERROR, ERROR')
            print('You need to load ALTAR result first!')
            sys.exit(0)


        if FigRange == 'All':
            indb = 0
            inde = self.AltarSamples.shape[1]
        else:
            indb = FigRange[0]
            inde = FigRange[1]

        BigS = self.AltarSamples
        Nsub = inde - indb # Number of histogramms to plot

        Npar = indb
        Nfig = (Nsub -1) / 9 + 1 # Number of figure
        Nrsub = Nsub % 9 # Number of "left over" histogramm

        if Nfig == 1:
            nrow,ncol = self.return_Nbsubfig(Nsub)
        else:
            nrow = 3; ncol = 3
    
        nsub_per_fig = nrow * ncol # Number of subplot per figures. 9 if several figures
        

        for i in range(Nfig): 
            if i == Nfig-1:
                nrow,ncol = self.return_Nbsubfig(Nrsub)
            axs = plt.subplots(nrow,ncol)
            for ii in range(nsub_per_fig):
                plt.subplot(nrow,ncol,ii)                
                plt.hist(BigS[:,indb+i*nsub_per_fig+ii],bins=50,normed=True)
                plt.xlim(xlim)
               
                if indb+i*nsub_per_fig+ii == inde -1:
                    break
        plt.show()
        plt.ion()

        return

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def return_Nbsubfig(self,Nsub):
        '''
        Give nrows and ncols for a given number of subfigs
        '''
        if Nsub == 1: 
            nrow=1; ncol=1
        elif Nsub == 2:
            nrow=2; ncol=1
        elif Nsub == 3:
            nrow=3; ncol=1
        elif Nsub == 4:
            nrow=2; ncol=2
        elif Nsub == 5:
            nrow=5; ncol=1
        elif Nsub == 6:
            nrow=3; ncol=2
        elif Nsub == 7:
            nrow=7; ncol=1
        elif Nsub == 8:
            nrow=4; ncol=2
        elif Nsub == 9:
            nrow=3; ncol=3

        return nrow, ncol


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def ComputeMw(self, mu = 3e10, Hist = False, ReOrganize = False):
        
        ''' 
        Compute Mw of ALTAR Solution
        '''
        

        print('-------------------------------------')
        print('--------- Compute Mo and Mw ---------')
        print('-------------------------------------')

        
        # Check something
        if not hasattr(self,'AltarMmean'):
            print('\n ERROR, ERROR')
            print('You need to load ALTAR result first!')
            sys.exit(0)

        if not hasattr(self,'slipNorm'):
            print('\n ERROR, ERROR')
            print('You need to compute the norm of slip first!')
            sys.exit(0)
        
        # Bricolage pour compter Big Bear a part
        NpBB = self.slv.faults[0].slip.shape[0]
        AreaBB = np.zeros((NpBB))
        AreaLA = np.zeros((self.slv.nSlip/2-NpBB))

        # Triangular patches, outdated...
        faults = self.slv.faults
        if  self.slv.faults[0].patchType == 'triangletent':
            k = 0
            for fault in faults:
                for p in range(len(fault.area_tent)):
                    Area[k] = round(fault.area_tent[p]*1e6/3)
                    k += 1
        
        # Rectangular patches
        elif  self.slv.faults[0].patchType in ['triangle','rectangle']:
            k = 0
            # Landers
            for fault in self.slv.faults[1::]:
                for p in range(len(fault.area)):
                    AreaLA[k] = round(fault.area[p]*1e6)
                    k += 1
           
            k = 0
            # Big Bear 
            fault = self.slv.faults[0]
            for p in range(len(fault.area)):
                 AreaBB[k] = round(fault.area[p]*1e6)
                 k += 1

        # Just check something     
        if (len(AreaLA) + len(AreaBB)) != len(self.slipNorm):
            print('\n ERROR, ERROR')
            print('Area vector and slipNorm vectors are not the same length!')
            print('Something fucked up somewhere')
            sys.exit(0)
            
        self.M0 = {'Big Bear':0, 'Landers': 0}      
        self.Mw = {'Big Bear':0, 'Landers': 0}
       
        if Hist:
            if ReOrganize:
                SlipBB = np.sqrt(self.AltarSamples[:,0:NpBB]**2 + self.AltarSamples[:,NpBB:2*NpBB]**2)

                info = self.slv.paramDescription     
                SlipLA = np.zeros((self.AltarSamples.shape[0],1))          
                for fault in self.slv.faults[1::]:
                    ibss = int(info[fault.name]['Strike Slip'].replace(' ','').partition('-')[0])
                    iess = int(info[fault.name]['Strike Slip'].replace(' ','').partition('-')[2])
                    ibds = int(info[fault.name]['Dip Slip'].replace(' ','').partition('-')[0])
                    ieds = int(info[fault.name]['Dip Slip'].replace(' ','').partition('-')[2])
                    tmp = np.sqrt(self.AltarSamples[:,ibss:iess]**2 + self.AltarSamples[:,ibds:ieds]**2)
                    
                    SlipLA = np.concatenate((SlipLA,tmp),axis=1)
                SlipLA = np.delete(SlipLA,0,axis=1)

            else:
                Np = self.slv.nSlip
                SlipBB = np.sqrt(self.AltarSamples[:,0:NpBB]**2 + self.AltarSamples[:,Np/2:Np/2+NpBB]**2)
                SlipLA = np.sqrt(self.AltarSamples[:,NpBB:Np/2]**2 + self.AltarSamples[:,Np/2+NpBB::]**2)
        
              
            M0BB = mu * SlipBB.dot(AreaBB) 
            M0LA = mu * SlipLA.dot(AreaLA) 
            self.M0['Big Bear'] = M0BB
            self.M0['Landers'] = M0LA

            self.Mw['Big Bear'] = (2./3) * np.log10(M0BB) - 6.07 
            self.Mw['Landers'] = (2./3) * np.log10(M0LA) - 6.07 

        else:
            return

        

#        if Hist:
#            if not hasattr(self,'AltarSamples'):
#                print('\n ERROR, ERROR')
#                print('You need to load ALTAR result to see Mw histogramms!')
#                return
#            else:




# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def PlotOneFault(self,faultname,plotList = ['strikeslip','dipslip'],figStart = 1):

        '''
        Plot a single fault segment
        arg:
        fault       : name of the fault to plot (string)
        '''
        
        print('-------------------------------------')
        print('--------- Plotting a fault ----------')
        print('-------------------------------------')

        # Check if ready
        if not self.NewM:
            print('\n ERROR, ERROR')
            print('You need to load ALTAR result in Slv object first!')
            sys.exit(0)

        faults = self.slv.faults
        for f in faults:
            if f.name == faultname:
                fault = f
                break

        if not 'fault' in locals():
            print('\n ERROR, ERROR')            
            print('Fault {} not found'.format(faultname))
            sys.exit(0)


        colorbar = True


        if  fault.patchType == 'triangletent':
            fignum = figStart
            for slipMode in plotList:
                gp = geoplt(fignum)
                gp.faulttrace(fault, color='r')
                gp.faultTents(fault, Norm=self.bounds[slipMode], colorbar=colorbar, slip=slipMode, method = 'scatter')
                fignum += 1

        else: #if fault.patchType == 'triangle':
            figNum = figStart
            for slipMode in plotList:
                gp = geoplt(figure=figNum)
                gp.faulttrace(fault, color='r')
                gp.faultpatches(fault, Norm=self.bounds[slipMode], colorbar=colorbar, slip=slipMode)
                figNum +=1
           
        plt.show()
        plt.ion()

        # All done
        return


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
    def PlotAllSegmentsSeparetly(self,plotList = ['strikeslip','dipslip']):
        '''
        Plot every segment but one per figure single fault segment
        
        arg:
        '''

        faults = self.slv.faults

        k = 1
        for fault in faults:
             self.PlotOneFault(fault.name,plotList = plotList,figStart = k)
             k += 10


        # All done
        return

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------        
    def writeSynth2File(self,name='',cosires=False,data2write='all'):
        '''
        Write GPS and TRI data to file
            * name : Add string to filenames
            * cosires : If True, write
        '''

        if not self.NewSynth:
            print('\n ERROR, ERROR')
            print('You need to compute new synthetics first!')
            sys.exit(0)

        if data2write == 'all':
            data2write == ['gps','insar','opticorr']

        
        datasets = self.datasets
        for data in datasets:
            if data.dtype == 'gps':
                if data.dtype in data2write:
                    fname = 'Synth_{}_{}.dat'.format(data.name,name)
                    with open(fname,'wb') as output:
                        for i in range(len(data.synth)):
                            line = '{}  {}  {}  {}  {}  {}  0.  0.  0.\n'.format(data.station[i],data.lon[i]-360,data.lat[i],
                            data.synth[i,0]*100, data.synth[i,1]*100, data.synth[i,2]*100)
                            output.write(line)


            elif data.dtype == 'insar':
                if data.dtype in data2write:
                    fout = open('xyz_{}_{}.xyz'.format(data.name,name), 'wb')
                    for i in range(data.lon.shape[0]):
                        fout.write('{} {} {} \n'.format(data.lon[i], data.lat[i], data.synth[i]*1000))
                    fout.close()

            elif data.dtype == 'opticorr':
                if data.dtype in data2write:
                    fname='synth_'+data.name+'_'+name
                    data.write2grd(fname,data='synth',interp=1000)
                    if cosires:
                        tmpN = data.north.copy()
                        tmpE = data.east.copy()
                        data.north -= data.north_synth
                        data.east -= data.east_synth
                        fname='res_'+data.name+'_'+name
                        data.write2grd(fname,data='data',interp=1000)
                        data.north = tmpN.copy()
                        data.east = tmpE.copy()




        # All done
        return
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------        
    def ComputeCm(self, ReOrganize = False):
        '''
        Compute Cm with ALTAR result
        '''

        print('-------------------------------------')
        print('------------ Compute Cm -------------')
        print('-------------------------------------')

        # Check something
        if not hasattr(self,'AltarSamples'):
            print('\n ERROR, ERROR')
            print('You need to load ALTAR result first!')
            sys.exit(0)
    
        if ReOrganize:
            theta = self.SavedSamples

        else:
            theta = self.AltarSamples
        
        Np = self.slv.nSlip /2
        Cm = np.zeros((Np,3),dtype='double')



        ss = theta[:,:Np]
        ds = theta[:,Np:2*Np]

        for p in range(Np):
            M = np.array([ss[:,p],ds[:,p]])
            cov = np.cov(M)
            Cm[p,:] = np.array([cov[0,0],cov[1,1],cov[0,1]])

        N = 0
        for f in self.slv.faults:
            Npf = f.slip.shape[0]
            f.Cm = Cm[N:N+Npf,:]
            N += Npf

        # All done
        return

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------        
    def writeResult2File(self,name=''):

        if len(self.slv.faults) == 7:
            num = {'Big_Bear': '0_',
                   'Homestead_Emerson_junction' : '3_',
                   'Johnson_valley' : '6_',
                   'Emerson_and_Camp-Rock' : '2_',
                   'Kickapoo' : '5_',
                   'Homestead_valley': '4_',
                   'Galway_lake': '1_'       
                   }
 
            fact = {'Big_Bear': 1.,
                   'Homestead_Emerson_junction' : 1.,
                   'Johnson_valley' : 1.,
                   'Emerson_and_Camp-Rock' : 1.,
                   'Kickapoo' : 1.,
                   'Homestead_valley': 1.,
                   'Galway_lake': 1.       
                   }


        elif len(self.slv.faults) == 9:
            num = {'Big_Bear': '0_',
                   'Homestead_Emerson_junction' : '5_',
                   'Johnson_valley' : '8_',
                   'Emerson_and_Camp-Rock' : '4_',
                   'Kickapoo' : '7_',
                   'Homestead_valley': '6_',
                   'Galway_lake': '2_',       
                   'Append_North': '1_',
                   'Append_South': '3_'
                   }

            fact = {'Big_Bear': 1.,
                   'Homestead_Emerson_junction' : 1.,
                   'Johnson_valley' : 1.,
                   'Emerson_and_Camp-Rock' : 1.,
                   'Kickapoo' : 1.,
                   'Homestead_valley': 1.,
                   'Galway_lake': 1.,       
                   'Append_North': 1.,
                   'Append_South': 1.
                   }

            indc = {'Big_Bear': [0,6,-1,15],
                   'Homestead_Emerson_junction' : [0,1,-1,-1],
                   'Johnson_valley' : [0,9,-1,26],
                   'Emerson_and_Camp-Rock' : [0,31,-1,-4],
                   'Kickapoo' : [0,1,-1,-1],
                   'Homestead_valley': [0,19,-1,-3],
                   'Galway_lake': [0,3,-1,-1],       
                   'Append_North': [0,15,-1,-2],
                   'Append_South': [0,7,-1,-1]
                   }


        elif len(self.slv.faults) == 10:
            num = {'Big_Bear': '0_',
                   'Homestead_Emerson_junction' : '5_',
                   'Johnson_valley' : '8_',
                   'Emerson_and_Camp-Rock' : '4_',
                   'Kickapoo' : '7_',
                   'Homestead_valley': '6_',
                   'Galway_lake': '2_',       
                   'Append_North': '1_',
                   'Append_South': '3_',
                   'Deep': '9_'
                   }

            fact = {'Big_Bear': 1.,
                   'Homestead_Emerson_junction' : 1.,
                   'Johnson_valley' : 1.,
                   'Emerson_and_Camp-Rock' : 1.,
                   'Kickapoo' : 1.,
                   'Homestead_valley': 1.,
                   'Galway_lake': 1.,       
                   'Append_North': 1.,
                   'Append_South': 1.,
                   'Deep': 1.
                   }


        for fault in self.slv.faults:   
            #if fault.patchType in ['rectangle','triangle']:
            fault.writeSlipDirection2File(num[fault.name]+fault.name+name+'.slipdir',scale='total',
                                            factor=0.1*fact[fault.name],ellipse=True,nsigma=2.4477) 
            fault.writePatches2File(num[fault.name]+fault.name+name+'.fault',add_slip='total')

            #elif fault.patchType == 'triangletent':
            #    fault.writeSlipDirection2File(num[fault.name]+fault.name+name+'.slipdir',scale='total',
            #                                    factor=0.1*fact[fault.name],ellipse=True,nsigma=2.4477) 
            #    fault.writePatches2File(num[fault.name]+fault.name+name+'.fault',add_slip='total')

                

                #fid = open(num[fault.name]+fault.name+name+'.fault_xyz','w')
                #for p in range(len(fault.patchll)):
                #    lon, lat, z = fault.getcenter(fault.patchll[p])
                #    slip = np.sqrt(fault.slip[p,0]**2+fault.slip[p,1]**2)
                #    fid.write('{}\t{}\t{}\n'.format(lon,lat,slip))
                #fid.close()
           
                #corners = [[fault.patchll[indc[fault.name][0]][0,0],fault.patchll[indc[fault.name][0]][0,1]],
                #           [fault.patchll[indc[fault.name][1]][1,0],fault.patchll[indc[fault.name][1]][1,1]],
                #           [fault.patchll[indc[fault.name][2]][2,0],fault.patchll[indc[fault.name][2]][2,1]],
                #           [fault.patchll[indc[fault.name][3]][3,0],fault.patchll[indc[fault.name][3]][3,1]]]
            
                #np.savetxt(num[fault.name]+fault.name+name+'.mask',corners,fmt='%2.10f')



        # All done
        return

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------  
    def writeTent2xyz(self,name=''):

        import csi.geodeticplot as geoplot

        num = {'Big_Bear': '0_',
               'Homestead_Emerson_junction' : '5_',
               'Johnson_valley' : '8_',
               'Emerson_and_Camp-Rock' : '4_',
               'Kickapoo' : '7_',
               'Homestead_valley': '6_',
               'Galway_lake': '2_',       
               'Append_North': '1_',
               'Append_South': '3_'
               }

        
        for fault in self.slv.faults:   
            gp = geoplot(figure=None)
            lon, lat, z, s = gp.faultTents(fault, slip='total', method='scatter',npoints=50)
            gp.close()
        
            arr = np.array([lon,lat,z,s]).T
            np.savetxt(num[fault.name]+fault.name+'.xyz',arr)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------    
    def adjustPolyPb(self):

        faults = self.slv.faults
        if len(faults)==7: 
            Galway_poly = copy.deepcopy(faults[-1].poly)
            Galway_polysol = copy.deepcopy(faults[-1].polysol)
            for k in Galway_polysol.keys():
                Galway_polysol[k] /= 7.      

        elif len(faults)==9: 
            Galway_poly = copy.deepcopy(faults[-1].poly)
            Galway_polysol = copy.deepcopy(faults[-1].polysol)
            for k in Galway_polysol.keys():
                Galway_polysol[k] /= 9. 
                
                
        elif len(faults)==10: 
            Galway_poly = copy.deepcopy(faults[-1].poly)
            Galway_polysol = copy.deepcopy(faults[-1].polysol)
            for k in Galway_polysol.keys():
                Galway_polysol[k] /= 10.  

        for fault in faults:
            fault.poly = Galway_poly
            fault.polysol = Galway_polysol
                
        return      


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------  
    def ComputeMwfpp(self,write2file = True):

        fpps = []
        Mos = []

        faults = copy.deepcopy(self.slv.faults)

        tmpf = copy.deepcopy(faults[1])
        for fault in faults[2:]:
            tmpf.addPatchesFromOtherFault(fault)
            tmpf.mu = np.append(tmpf.mu,fault.mu)
            tmpf.area = np.append(tmpf.area,fault.area)

        if self.slv.nSlip == 412:
            indss = [17,206]
            indds = [206+17,412]
        elif self.slv.nSlip == 408:
            indss = [17,204]
            indds = [204+17,408]

        fpp = faultpp('tmp',tmpf,Mu=tmpf.mu,utmzone=11,samplesh5=self.altarfile)
        fpp.h5_init(indss=indss,indds=indds)
        fpp.h5_finalize()
        fpp.computeMomentTensor()
        Mola=fpp.computeScalarMoment()
        Mwla=fpp.computeMagnitude(plotHist='./')      
            

        ## Big Bear
        fault = faults[0]

        fpp = faultpp(fault.name,fault,Mu=fault.mu,utmzone=11,samplesh5=self.altarfile)
        fpp.h5_init(indss=[0,17],indds=[self.slv.nSlip/2,self.slv.nSlip/2+17])
        fpp.h5_finalize()
        fpp.computeMomentTensor()
        Mobb=fpp.computeScalarMoment()
        Mwbb=fpp.computeMagnitude()


        self.M0fpp = {'Big Bear':Mobb, 'Landers': Mola}      
        self.Mwfpp = {'Big Bear':Mwbb, 'Landers': Mwla}


        if write2file:
            fout = open('Magnitude.txt','w')
            fout.write('Big Bear \tLanders\n')
            fout.write('{} \t{}'.format(Mwbb.mean(),Mwla.mean()))

