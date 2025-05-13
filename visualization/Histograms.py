
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import f as fff
from scipy import stats
from scipy.stats import mannwhitneyu
import os

class Histograms():
    def __init__(self, config):
        self.config = config
        self.histogram_max_y_all = config.feature_extraction.histogram_max_y
        self.histogram_max_x_all = config.feature_extraction.histogram_max_x
        self.histogram_max_y_channels = config.feature_extraction.histogram_max_y_channels
        self.histogram_max_x_channels = config.feature_extraction.histogram_max_x_channels
        self.linewidth = 0.3
        self.alpha_histogram = 0.3
        self.bins = config.feature_extraction.bins
    
        self.control_str = config.data.control
        self.disease_str = config.data.disease


        self.output_path = config.output.loc
        os.makedirs(self.output_path, exist_ok=True)
        self.output_dir = self.output_path
        os.makedirs(os.path.join(self.output_dir), exist_ok=True)
        self.output_sub_dir = config.feature_extraction.method
        os.makedirs(os.path.join(self.output_dir,self.output_sub_dir), exist_ok=True)

        return
    
    def get_all_channel_histogram(self,cont_feat_flatten,disease_feat_flatten,feature_labels):
        '''Given disease and control inputs, plot histogram
        disease_feat_flatten and control_feat_flatten: Arrays with shape (pat, feature, :) where : is the squished channels of the image'''
        #this will save a final figure of 4, the disease and controls as histograms, some with lines and some with overlaying histograms to see differences
        #stats current are the mean of both groups, manwhitney test and t-test. The best output depends on shape.

        fig, axes = plt.subplots(1,4,figsize=(20, 5))
        for f in range(len(feature_labels)):
            feature_name = feature_labels[f]
            print(feature_name)
            kwargs = dict(histtype='stepfilled', alpha=self.alpha_histogram, bins=self.bins)

            for pat in range(cont_feat_flatten.shape[0]):
                if pat == 0:
                    axes[0].hist((cont_feat_flatten[pat,f,:]), color = 'b',label=self.control_str, **kwargs)
                    axes[1].hist((cont_feat_flatten[pat,f,:]), color = 'b',histtype="step",linewidth=self.linewidth,label=self.control_str,bins=self.bins)
                    axes[2].hist((cont_feat_flatten[pat,f,:]), color = 'b',histtype="step",linewidth=self.linewidth,label=self.control_str,bins=self.bins)
                    axes[3].hist((cont_feat_flatten[pat,f,:]), color = 'b',label=self.control_str, **kwargs)

                else:
                    axes[0].hist((cont_feat_flatten[pat,f,:]), color = 'b', **kwargs)
                    axes[1].hist((cont_feat_flatten[pat,f,:]), color = 'b',histtype="step",linewidth=self.linewidth,bins=self.bins)
                    axes[2].hist((cont_feat_flatten[pat,f,:]), color = 'b',histtype="step",linewidth=self.linewidth,bins=self.bins)
                    axes[3].hist((cont_feat_flatten[pat,f,:]), color = 'b', **kwargs)

            for pat in range(disease_feat_flatten.shape[0]):
                if pat == 0:
                    axes[0].hist((disease_feat_flatten[pat,f,:]), color = 'r', label=self.disease_str, **kwargs)
                    axes[1].hist((disease_feat_flatten[pat,f,:]), color = 'r',histtype="step",linewidth=self.linewidth,label=self.disease_str,bins=self.bins)
                    axes[2].hist((disease_feat_flatten[pat,f,:]), color = 'r',label=self.disease_str,**kwargs)
                    axes[3].hist((disease_feat_flatten[pat,f,:]), color = 'r',histtype="step",linewidth=self.linewidth,label=self.disease_str,bins=self.bins)

                else:
                    axes[0].hist((disease_feat_flatten[pat,f,:]), color = 'r', **kwargs)
                    axes[1].hist((disease_feat_flatten[pat,f,:]), color = 'r',histtype="step",linewidth=self.linewidth,bins=self.bins)
                    axes[2].hist((disease_feat_flatten[pat,f,:]), color = 'r',**kwargs)
                    axes[3].hist((disease_feat_flatten[pat,f,:]), color = 'r',histtype="step",linewidth=self.linewidth,bins=self.bins)


            vals_control=cont_feat_flatten[:,f,:]
            vals_disease=disease_feat_flatten[:,f,:]
            t_stat, p_value = stats.ttest_ind(vals_disease.flatten(), vals_control.flatten())
            man_stat, p_man = mannwhitneyu(vals_disease.flatten(), vals_control.flatten(), alternative='two-sided')
            

            STR="p="+str(p_value)

            if p_value<0.05:
                STR="t-test p="+str((p_value))+"**"+""
                STRmean="m_cont="+str(round(vals_control.mean(),3))
                STRmean_d="m_disease="+str(round(vals_disease.mean(),3))
            else:
                STR="t-test p="+str((p_value))
                STRmean="m_cont="+str(round(vals_control.mean(),3))
                STRmean_d="m_disease="+str(round(vals_disease.mean(),3))       
            
            man_stat, p_man = mannwhitneyu(vals_disease.flatten(), vals_control.flatten(), alternative='two-sided')
            if p_man<0.05:
                STR_MAN = "man p ="+str((p_man))+"**"+""                
            else:
                STR_MAN = "man p ="+str((p_man))                

            man_stat, p_man = mannwhitneyu(vals_disease.flatten(), vals_control.flatten(), alternative='two-sided')

            axes[0].text(0.05, 0.99, STR, horizontalalignment='left', verticalalignment='top',transform=axes[0].transAxes)
            axes[0].text(0.05, 0.95, STR_MAN, horizontalalignment='left', verticalalignment='top',transform=axes[0].transAxes)

            axes[0].text(0.05, 0.91, STRmean, horizontalalignment='left', verticalalignment='top',transform=axes[0].transAxes)
            axes[0].text(0.05, 0.87, STRmean_d, horizontalalignment='left', verticalalignment='top',transform=axes[0].transAxes)

            axes[0].set_title(feature_name, fontsize=10, pad=0)
            # plt.show()
            axes[0].legend()
            axes[1].legend()
            axes[2].legend()
            axes[3].legend()

            plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,feature_name+'_nocrop_histogram'))

            axes[0].set_xlim(0,self.histogram_max_x_all)
            axes[0].set_ylim(0,self.histogram_max_y_all)
            axes[1].set_xlim(0,self.histogram_max_x_all)
            axes[1].set_ylim(0,self.histogram_max_y_all)
            axes[2].set_xlim(0,self.histogram_max_x_all)
            axes[2].set_ylim(0,self.histogram_max_y_all)
            axes[3].set_xlim(0,self.histogram_max_x_all)
            axes[3].set_ylim(0,self.histogram_max_y_all)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,feature_name+'_histogram'))

        plt.close()
        return
    
    def get_individual_channel_histogram(self,cont_feat_flatten,disease_feat_flatten,feature_labels):
        '''Given disease and control inputs, plot histogram
        disease_feat_flatten and control_feat_flatten: Arrays with shape (pat, feature, channel, :) where : is the squished channels of the image'''
        #this will save a final figure of all 5 channels with the disease and controls as histograms
        #First row overall then second, third, 4th zooming in

        for f in range(len(feature_labels)):
            fig, axes = plt.subplots(4,cont_feat_flatten.shape[2],figsize=(20,25))

            #loop in number of channels
            for c in range((cont_feat_flatten.shape[2])):
                print(c)
                feature_name = feature_labels[f]
                print(feature_name)
                kwargs = dict(histtype='stepfilled', alpha=self.alpha_histogram, bins=self.bins)
                
                for pat in range(cont_feat_flatten.shape[0]):
                    if pat == 0:
                        axes[0][c].hist((cont_feat_flatten[pat,f,c,:]), color = 'b',label=self.control_str, **kwargs)
                        axes[1][c].hist((cont_feat_flatten[pat,f,c,:]), color = 'b',histtype="step",linewidth=self.linewidth,label=self.control_str,bins=self.bins)
                        axes[2][c].hist((cont_feat_flatten[pat,f,c,:]), color = 'b',histtype="step",linewidth=self.linewidth,label=self.control_str,bins=self.bins)
                        axes[3][c].hist((cont_feat_flatten[pat,f,c,:]), color = 'b',label=self.control_str, **kwargs)

                    else:
                        axes[0][c].hist((cont_feat_flatten[pat,f,c,:]), color = 'b', **kwargs)
                        axes[1][c].hist((cont_feat_flatten[pat,f,c,:]), color = 'b',histtype="step",linewidth=self.linewidth,bins=self.bins)
                        axes[2][c].hist((cont_feat_flatten[pat,f,c,:]), color = 'b',histtype="step",linewidth=self.linewidth,bins=self.bins)
                        axes[3][c].hist((cont_feat_flatten[pat,f,c,:]), color = 'b', **kwargs)

                for pat in range(disease_feat_flatten.shape[0]):
                    if pat == 0:
                        axes[0][c].hist((disease_feat_flatten[pat,f,c,:]), color = 'r', label=self.disease_str, **kwargs)
                        axes[1][c].hist((disease_feat_flatten[pat,f,c,:]), color = 'r',histtype="step",linewidth=self.linewidth,label=self.disease_str,bins=self.bins)
                        axes[2][c].hist((disease_feat_flatten[pat,f,c,:]), color = 'r', label=self.disease_str, **kwargs)
                        axes[3][c].hist((disease_feat_flatten[pat,f,c,:]), color = 'r',histtype="step",linewidth=self.linewidth,label=self.disease_str,bins=self.bins)
                    else:
                        axes[0][c].hist((disease_feat_flatten[pat,f,c,:]), color = 'r', **kwargs)
                        axes[1][c].hist((disease_feat_flatten[pat,f,c,:]), color = 'r',histtype="step",linewidth=self.linewidth,bins=self.bins)
                        axes[2][c].hist((disease_feat_flatten[pat,f,c,:]), color = 'r', **kwargs)
                        axes[3][c].hist((disease_feat_flatten[pat,f,c,:]), color = 'r',histtype="step",linewidth=self.linewidth,bins=self.bins)
 
                vals_control=cont_feat_flatten[:,f,c,:]
                vals_disease=disease_feat_flatten[:,f,c,:]
                t_stat, p_value = stats.ttest_ind(vals_disease.flatten(), vals_control.flatten())
                man_stat, p_man = mannwhitneyu(vals_disease.flatten(), vals_control.flatten(), alternative='two-sided')

                axes[0][c].set_title("channel "+str(c), fontsize=10, pad=0)
                axes[0][c].legend()

                axes[1][c].set_title("channel "+str(c), fontsize=10, pad=0)
                axes[1][c].legend()
                axes[1][c].set_xlim(0,self.histogram_max_x_channels)
                axes[1][c].set_ylim(0,self.histogram_max_y_channels)
                axes[1][c].legend()

                axes[2][c].set_title("channel "+str(c), fontsize=10, pad=0)
                axes[2][c].legend()
                axes[2][c].set_xlim(0,self.histogram_max_x_channels)
                axes[2][c].set_ylim(0,self.histogram_max_y_channels)
                axes[2][c].legend()

                axes[3][c].set_title("channel "+str(c), fontsize=10, pad=0)
                axes[3][c].legend()
                axes[3][c].set_xlim(0,self.histogram_max_x_channels)
                axes[3][c].set_ylim(0,self.histogram_max_y_channels)
                axes[3][c].legend()

                if p_value<0.05:
                    STR="t-test p="+str((p_value))+"**"+""
                    STRmean="m_cont="+str(round(vals_control.mean(),3))
                    STRmean_d="m_disease="+str(round(vals_disease.mean(),3))                
                else:
                    STR="t-test p="+str((p_value))
                    STRmean="m_cont="+str(round(vals_control.mean(),3))
                    STRmean_d="m_disease="+str(round(vals_disease.mean(),3))

                if p_man<0.05:
                    STR_MAN = "man p ="+str((p_man))+"**"+""                
                else:
                    STR_MAN = "man p ="+str((p_man))                

                axes[0][c].text(0.05, 0.99, STR, horizontalalignment='left', verticalalignment='top',transform=axes[0][c].transAxes)
                axes[0][c].text(0.05, 0.95, STRmean, horizontalalignment='left', verticalalignment='top',transform=axes[0][c].transAxes)
                axes[0][c].text(0.05, 0.91, STRmean_d, horizontalalignment='left', verticalalignment='top',transform=axes[0][c].transAxes)
                axes[0][c].text(0.05, 0.87, STR_MAN, horizontalalignment='left', verticalalignment='top',transform=axes[0][c].transAxes)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,feature_name+'_histogram_PERCHANNEL'))
            plt.close()
        
        return