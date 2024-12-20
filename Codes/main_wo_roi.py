import argparse
import os
import torch
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import phenocamdata
from trainer_wo_roi import Trainer # To train the model without ROI
from utils import make_folder

def main(args):
    # For fast training
    if args.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = True
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    # Load data
    print("Start loading data...")
    train_data = phenocamdata(args.image_path, args.site_name, args.image_height, args.image_width, args.roi_id, 
                             args.file_type, is_train=True, data_vol=args.train_data_vol_per)
    test_data = phenocamdata(args.image_path, args.site_name, args.image_height, args.image_width, args.roi_id, 
                             args.file_type, is_train=False, data_vol=1)
    print("Length of Train Dataset:" + str(len(train_data)))
    print("Length of Test Dataset:" + str(len(test_data)))
    # Load ROI (ROI is same for train and test data)
    roi = train_data.load_roi() 
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Retrieve current working directory
    curr_wd = os.path.dirname(__file__)
    
    # Create folder to save plots badsed on available data
    args.save_plot_path = os.path.join(curr_wd, args.save_plot_path)
    make_folder(args.save_plot_path, args.site_name +"_" + args.roi_id)
    args.save_plot_path = os.path.join(args.save_plot_path, args.site_name +"_" + args.roi_id)
    
    #plot_gcc_label_dist(os.path.join(args.save_plot_path, "gcc_label_distribution_train_data.png"), train_data.dataset["gcc_label"])
    #plot_gcc_label_dist(os.path.join(args.save_plot_path, "gcc_label_distribution_test_data.png"), test_data.dataset["gcc_label"])
    #train_data.plot_real_image(nrows=1, ncols=6, 
    #                           save_file_name = os.path.join(args.save_plot_path, "sample_real_images_train"+str(date.today().strftime("%m-%d-%Y"))+".png"))
    #test_data.plot_real_image(nrows=1, ncols=6, 
    #                          save_file_name = os.path.join(args.save_plot_path, "sample_real_images_test"+str(date.today().strftime("%m-%d-%Y"))+".png"))
    
    # Create folder to save models, sample synthetic images and logs
    folder_name = args.model + "_" + args.adv_loss + "_" + args.version
    args.model_save_path = os.path.join(curr_wd, args.model_save_path, args.site_name+"_"+args.roi_id)
    args.sample_path = os.path.join(curr_wd, args.sample_path, args.site_name+"_"+args.roi_id )
    args.log_path = os.path.join(curr_wd, args.log_path, args.site_name+"_"+args.roi_id)
    make_folder(args.model_save_path, folder_name)
    make_folder(args.sample_path, folder_name)
    make_folder(args.log_path, folder_name)
    if args.load_model_path is not None:
        args.load_model_path = os.path.join(curr_wd,args.load_model_path)
    # Create a file to log output
    exec_summary_file_name = "Execution_summary_" + str(date.today().strftime("%m-%d-%Y"))+ ".txt"
    with open(os.path.join(args.log_path, folder_name, exec_summary_file_name), 'w') as txt_file_object:
        txt_file_object.write("Execution Summary of GAN:")
        txt_file_object.write("\nSpectral_G: %s; Specatral_D: %s; Batch_Norm_G: %s; Batch_Norm_D:%s"%(args.spectral_G, args.spectral_D, args.batch_norm_G, args.batch_norm_D))
        txt_file_object.write("\nlr_G: %s; lr_D: %s; beta1: %s; beta2:%s"%(args.lr_G, args.lr_D, args.beta1, args.beta2))
        txt_file_object.write("\nD_iters: %d; G_iters: %d; latent_dim: %d"%(args.D_iters, args.G_iters, args.latent_dim))
        txt_file_object.close()

    # Train GAN
    print("Start training GAN...")
    trainer = Trainer(train_loader, test_data, roi, args, device)
    trainer.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training GAN on PhenoCam Images")
    # specify to enable cuda
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu_device', default="1,2,3,0", help="GPU devices to be used")
    
    # site and image detainand ROI specifications
    parser.add_argument('--site_name', type=str, default='NEON.D01.HARV.DP1.00033', help='PhenoCam Site ID')
    parser.add_argument('--roi_id', type=str, default='DB_1000', help = "ROI ID")
    parser.add_argument('--n_channels', type=int, default=3, help = 'number of channels in the image')
    parser.add_argument('--image_height', type=int, default=480)
    parser.add_argument('--image_width', type=int, default=648)
    parser.add_argument('--file_type', type=str, default='/*.jpg')
    
    # path details
    parser.add_argument('--image_path', type=str, default='/research/iprobe-paldebas/Research_Work/GAN (Phenology Images)/phenocamdata/')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--save_plot_path', type=str, default='plots')
    
    # model parameters 
    parser.add_argument('--model', type=str, default='DCGAN', choices=['DCGAN','DCGAN_1'])
    parser.add_argument('--adv_loss', type=str, default='HingeGAN', choices=['HingeGAN'])
    parser.add_argument('--version', type=str, default= 'TEST')
    parser.add_argument('--spectral_G', default=True, help='Whetherto use spectral normalization in generator')
    parser.add_argument('--spectral_D', default=True, help='Whether to use spectral normalization in discriminator')
    parser.add_argument('--batch_norm_G', default=True, help='Whether to use batch normalization in generator')
    parser.add_argument('--batch_norm_D', default=True, help='Whether to use batch normalization in discriminator')
    parser.add_argument('--bias_G', default=False, help='Bias in generator')
    parser.add_argument('--bias_D', default=False, help='Bias in discriminator')
    
    
    # details of training paramaters
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for training")
    parser.add_argument('--num_epoch', type=int, default=1000, help="number of iterations")
    parser.add_argument('--D_iters', type=int, default=1, help='Number of iterations of D')
    parser.add_argument('--G_iters', type=int, default=1, help='Number of iterations of G.')
    parser.add_argument('--latent_dim', type=int, default=128, help="latent dimension for generator")
    parser.add_argument('--embedding_dim', type=int, default=30, help="embedding dimension (not used in the model)")
    parser.add_argument('--num_workers', type=int, default=8)
    
    # details of hyperparameters
    parser.add_argument('--lr_D', type=float, default=.0001, help='Discriminator learning rate')
    parser.add_argument('--lr_G', type=float, default=.00005, help='Generator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam betas[0]')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight. Helps convergence but leads to artifacts in images, not recommended.')
    parser.add_argument('--decay_D', type=float, default=0.0, help='Decay to apply to learning rate each cycle. decay^n_iter gives the final lr.')    
    parser.add_argument('--decay_G', type=float, default=0.0, help='Decay to apply to learning rate each cycle. decay^n_iter gives the final lr.')    
    #parser.add_argument('--lambda_gp', type=float, default=10)
    
    # Other parameters
    parser.add_argument('--log_step', type=int, default=1, help='To show log')
    parser.add_argument('--sample_step', type=int, default=5, help='To save samples of generated images')
    parser.add_argument('--model_save_step', type=int, default=5, help='To save model')
    parser.add_argument('--load_model', type=int, default=None)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--train_data_vol_per', type=float, default=1.0, help='Percentage of training data to be used')
    parser.add_argument('--num_samples', type=int, default=20, help='For display purposes')
    parser.add_argument('--n_rows', type=int, default=4, help='For display purposes')
    parser.add_argument('--n_cols', type=int, default=5, help='For display purposes')
    parser.add_argument('--plot_figsize', default=(22,20), help='For display purposes') 
    parser.add_argument('--gcc_bins', default=[0.30, 0.34, 0.36, 0.38, 0.40, 0.48])  # For testing purpose
    
    args = parser.parse_args()
    main(args)