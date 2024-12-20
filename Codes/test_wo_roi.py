import os
import argparse
import numpy as np
from datetime import date
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from dataset import phenocamdata
from utils import *
import seaborn as sns

class Tester():
    def __init__(self, test_data, test_loader, roi, args, device):
        self.test_data = test_data
        self.test_loader = test_loader
        self.roi = roi
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.device = device
        self.latent_dim = args.latent_dim
        self.embedding_dim = args.embedding_dim
        self.args = args
        self.model_save_path = args.model_save_path
        self.synthetic_img_path = args.synthetic_img_path
        
    def load_trained_model(self,trained_model):
        if self.args.model == "DCGAN":
            #from DCGAN_adjusted_v13 import Generator
            from DCGAN_adjusted_v13_wo_ROI import Generator
    
        self.G = Generator(self.image_height, self.image_width, self.latent_dim, self.embedding_dim,
                           self.args.spectral_G, self.args.batch_norm_G, self.args.bias_G).to(self.device)
        self.G = torch.nn.DataParallel(self.G)
        self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_G.pth'.format(trained_model))))
        
    def generate_synthetic_image_samples(self, trained_model):
        computed_gcc = []
        predicted_rcc = []
        ssim_score_list = []
        modified_ssim_score_list = []

        #ssim_val = np.empty([len(self.test_data.dataset),1])
        #ms_ssim_val = np.empty([len(self.test_data.dataset),1])
        make_folder(self.synthetic_img_path, "Epoch_"+str(trained_model))
        num_steps = len(self.test_loader)
        data_itr = iter(self.test_loader)
        self.G.eval()
        self.roi = Variable(self.roi.to(self.device), requires_grad= False)
        for step in range(0, num_steps):
            real_images, values, file_name = next(data_itr)
            current_batch_size = real_images.size(0)
            z = Variable(torch.randn(current_batch_size, self.latent_dim).to(self.device), requires_grad = False)
            z_values = Variable(values.view(current_batch_size,1).type(torch.FloatTensor).to(self.device), requires_grad = False)
            synthetic_images = self.G(z, z_values).detach()
            #ssim_val = np.concatenate(ssim_val, calculate_ssim(denorm(real_images), denorm(synthetic_images)))
            #ms_ssim_val = np.concatenate(ms_ssim_val, calculate_ms_ssim(denorm(real_images), denorm(synthetic_images)))
            for i in range(0, current_batch_size):
                test_img = denorm(real_images.data[i]).to("cpu")
                synthetic_img = denorm(synthetic_images[i].to("cpu"))
                plt.imsave("test.jpg", tensor_to_PIL(synthetic_img))
                save_image(synthetic_img, 
                       os.path.join(self.synthetic_img_path, "Epoch_"+str(trained_model), 
                                    'synthetic_{}_{}.jpg'.format(file_name[i], self.args.batch_size*step+i)))
                gcc = calculate_gcc(synthetic_img, self.roi.to("cpu"))
                rcc = calculate_rcc(synthetic_img, self.roi.to("cpu"))
                ssim_score = calculate_ssim_score(np.array(tensor_to_PIL(test_img)), np.array(tensor_to_PIL(synthetic_img)))
                ssim_score_modfied = self.synthetic_image_ssim_score(synthetic_img, values[i].item())
                computed_gcc.append(gcc)
                predicted_rcc.append(rcc)
                ssim_score_list.append(ssim_score) 
                modified_ssim_score_list.append(ssim_score_modfied)
        np.save(os.path.join(self.synthetic_img_path, "computed_gcc_epoch_"+str(trained_model)+".npy"), computed_gcc)
        np.save(os.path.join(self.synthetic_img_path, "predicted_rcc_epoch_"+str(trained_model)+".npy"), predicted_rcc)
        np.save(os.path.join(self.synthetic_img_path, "ssim_epoch_"+str(trained_model)+".npy"), ssim_score_list)
        np.save(os.path.join(self.synthetic_img_path, "modified_ssim_epoch_"+str(trained_model)+".npy"), modified_ssim_score_list)
        
        #np.save(os.path.join(self.synthetic_img_path, "ssim_new_epoch_"+str(trained_model)+".npy"), ssim_val)
        #np.save(os.path.join(self.synthetic_img_path, "ms_ssim_epoch_"+str(trained_model)+".npy"), ms_ssim_val)
                    
    def plot_ssim_histogram(self, trained_model):
        data = np.load(os.path.join(self.synthetic_img_path, "ssim_epoch_"+str(trained_model)+".npy"))
        fig = sns.histplot(data, color = 'gray')
        fig.set_xlabel("SSIM", fontsize=20)
        fig.set_ylabel("Count", fontsize=20)
        plt.savefig(os.path.join(self.synthetic_img_path, "ssim_histogram_epoch_"+str(trained_model)+".png"))
        plt.close()
  
    def synthetic_image_ssim_score(self, synthetic_img, gcc_value):
        ssim_score_list = []
        df = self.test_data.dataset[self.test_data.dataset["gcc_rounded"]==gcc_value]
        synthetic_img = np.array(tensor_to_PIL(synthetic_img))
        if df.empty:
            print("No test data available.")
            return 0
        #elif len(df.index) > 10:
        #    df = df.sample(n=10).reset_index()
        else:
            df = df.reset_index()
        for ind in df.index:
            real_img = np.array(tensor_to_PIL(denorm(self.test_data.__getitem__(df["index"][ind])[0])))
            ssim_score_list.append(calculate_ssim_score(real_img,synthetic_img))
        return max(ssim_score_list)

        
        
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    device = torch.device('cuda')
        
    # Create folder
    curr_wd = os.path.dirname(__file__)
    folder_name = args.model + "_" + args.adv_loss + "_" + args.version
    args.synthetic_img_path = os.path.join(curr_wd, args.synthetic_img_path, args.site_name+"_"+args.roi_id)
    args.model_save_path = os.path.join(curr_wd, args.model_save_path, args.site_name+"_"+args.trained_model_roi, folder_name)
    make_folder(args.synthetic_img_path, folder_name)
    args.synthetic_img_path = os.path.join(args.synthetic_img_path, folder_name) 
   
    test_data = phenocamdata(args.image_path, args.site_name, args.image_height, args.image_width, args.roi_id, 
                             args.file_type, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    roi = test_data.load_roi()
    tester = Tester(test_data, test_loader, roi, args, device)
    for trained_model in args.trained_models:
        tester.load_trained_model(trained_model)
        tester.generate_synthetic_image_samples(trained_model)
        #tester.calculate_rmse_gcc_rcc(trained_model, val_summary_file_name)
        #tester.plot_ssim_histogram(trained_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SSIM score calculation")
    
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu_device', default="4,5,6,7", help="GPU devices to be used")
    
    # Site and ROI specifications
    parser.add_argument('--site_name', type=str, default='NEON.D01.HARV.DP1.00033')
    parser.add_argument('--roi_id', type=str, default='DB_1000')
    parser.add_argument('--image_height', type=int, default=480)
    parser.add_argument('--image_width', type=int, default=648)
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--file_type', type=str, default='/*.jpg')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=100)
    
    # path details
    parser.add_argument('--image_path', type=str, default='/research/iprobe-paldebas/Research_Work/GAN (Phenology Images)/phenocamdata/')
    parser.add_argument('--synthetic_img_path', type=str, default='synthetic_test_images')
    parser.add_argument('--model_save_path', type=str, default='models')
    
    # Model parameters 
    parser.add_argument('--model', type=str, default='DCGAN', choices=['DCGAN', 'DCGAN_1'])
    parser.add_argument('--adv_loss', type=str, default='HingeGAN', choices=['HingeGAN'])
    parser.add_argument('--version', type=str, default= 'v13_SN_SA_3_orthogonal_wo_roi_retest_0.5')
    parser.add_argument('--spectral_G', default=True, help='Whether to use spectral normalization in generator')
    parser.add_argument('--spectral_D', default=True, help='Whether to use spectral normalization in discriminator')
    parser.add_argument('--batch_norm_G', default=True, help='Whether to use batch normalization in generator')
    parser.add_argument('--batch_norm_D', default=True, help='Whether to use batch normalization in discriminator')
    parser.add_argument('--bias_G', default=False, help='Bias in generator')
    parser.add_argument('--bias_D', default=False, help='Bias in discriminator')
    parser.add_argument('--latent_dim', type=int, default=128, help="latent dimension for generator")
    parser.add_argument('--embedding_dim', type=int, default=30, help="embedding dimension for class label")
        
    parser.add_argument('--trained_models', nargs='+', type=int, default=[975])
    parser.add_argument('--trained_model_roi', type=str, default="DB_1000")
    args = parser.parse_args()
    main(args)     
 