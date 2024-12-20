import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm
#from spectral import SpectralNorm

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)+ 1e-8)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X C X (N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy = torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out


def deconv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, spectral_norm=False, batch_norm=False, pixel_norm = False, activation = 'ReLU', self_attn = False):
    if spectral_norm:
        layer = [SpectralNorm(torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding, bias=bias))]
    else:
        layer = [torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)]
    if pixel_norm:
        layer += [PixelNorm()]
    elif batch_norm:
        layer += [torch.nn.BatchNorm2d(out_channels)]
    if activation == "ReLU":
        layer += [torch.nn.ReLU(True)]
    elif activation == "LeakyReLU":
        layer += [torch.nn.LeakyReLU(0.1, inplace=True)]
    elif activation == "Tanh":
        layer += [torch.nn.Tanh()]
    if self_attn == True:
        layer += [Self_Attn(out_channels, 'relu')]
    return layer   

def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, spectral_norm=False, batch_norm=False, activation = 'LeakyReLU', dropout=None, self_attn = False):
    if spectral_norm:
        layer = [SpectralNorm(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))]
    else:
        layer = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
    if batch_norm:
        layer += [torch.nn.BatchNorm2d(out_channels)]
    if activation == "LeakyReLU":
        layer += [torch.nn.LeakyReLU(0.1, inplace=True)]
    if dropout is not None:
        layer += [torch.nn.Dropout(p=dropout)]
    if self_attn == True:
        layer += [Self_Attn(out_channels, 'relu')]
    return layer  

class Generator(nn.Module):
    """Generator."""

    def __init__(self, img_height=960, img_width=1296, latent_dim=128, embedding_dim=50, spectral_G = True, batch_norm_G=False, bias_G = True):
        super(Generator, self).__init__()
        self.embedding_dim = embedding_dim
        self.img_height = img_height
        self.img_width = img_width
        self.spectral_norm = spectral_G
        self.batch_norm = batch_norm_G
        self.bias = bias_G
        
        self.label_embedding = nn.Sequential(
            nn.Linear(1, int(latent_dim)),
            nn.ReLU(inplace=True)
        )
        
        self.roi_reshape = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.img_height*self.img_width, int(latent_dim)),
            nn.ReLU(inplace=True)
        )
        
        model = deconv_layer(latent_dim * 3, 512, kernel_size=(5,6), bias=self.bias, 
                             spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="ReLU")
        
        model += deconv_layer(512, 256, kernel_size=4, stride=2, padding=1, output_padding=(0,1), bias=self.bias,
                              spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="ReLU")
        
        model += deconv_layer(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(0,1), bias=self.bias, 
                              spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="ReLU",self_attn=False)
        
        model += deconv_layer(128, 128, kernel_size=4, stride=2, padding=1, bias=self.bias,
                              spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="ReLU", self_attn=False)
        
        model += deconv_layer(128, 64, kernel_size=4, stride=2, padding=1, bias=self.bias,
                              spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="ReLU",self_attn=False)
        
        model += deconv_layer(64, 64, kernel_size=6, stride=3, padding=2, output_padding=1, bias=self.bias, 
                              spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="ReLU", self_attn=False)
        
        #model += deconv_layer(16, 16, kernel_size=4, stride=2, padding=1, bias=self.bias, 
        #                      spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="ReLU")
        
        model += deconv_layer(64, 3, kernel_size=4, stride=2, padding=1, bias=self.bias, 
                              spectral_norm=False, batch_norm= False, activation="Tanh")
        
        self.model = torch.nn.Sequential(*model)
		        
    def forward(self, z, roi, z_values):
        batch_size = z.size(0)
        latent_dim = z.size(1)
        z = z.view(batch_size, latent_dim, 1, 1)
        roi = self.roi_reshape(roi)
        roi = roi.view(batch_size, int(latent_dim), 1, 1)
        embed_label = self.label_embedding(z_values)
        embed_label = embed_label.view(batch_size, int(latent_dim), 1, 1)
        gen_input = torch.cat((z, roi, embed_label), dim=1)
        out=self.model(gen_input)
        return out


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, img_height=960, img_width=1296, embedding_dim=50, spectral_D=False, batch_norm_D=True, bias_D=True, sigmoid = False):
        super(Discriminator, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.spectral_norm = spectral_D
        self.batch_norm = batch_norm_D
        self.bias = bias_D
        
        self.label_embedding = nn.Sequential(
            nn.Linear(1, img_height*img_width),
            #nn.BatchNorm1d(img_height*img_width),
            nn.LeakyReLU(0.1, inplace=True)
        )

        #self.label_input = nn.Sequential(
            #nn.BatchNorm2d(1),
         #   nn.LeakyReLU(0.1, inplace=True)

        #)

        model = conv_layer(5, 32, kernel_size=4, stride=2, padding=1, bias=self.bias, 
                           spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="LeakyReLU")
        model += conv_layer(32, 64, kernel_size=4, stride=2, padding=1, bias=self.bias, 
                            spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="LeakyReLU")
        model += conv_layer(64, 64, kernel_size=4, stride=2, padding=1, bias=self.bias, 
                            spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="LeakyReLU")
        model += conv_layer(64, 128, kernel_size=4, stride=2, padding=1, bias=self.bias, 
                            spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="LeakyReLU")
        model += conv_layer(128, 128, kernel_size=4, stride=2, padding=1, bias=self.bias, 
                            spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="LeakyReLU", self_attn=False)
        model += conv_layer(128, 256, kernel_size=4, stride=2, padding=1, bias=self.bias,
                            spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="LeakyReLU", self_attn=False)
        model += conv_layer(256, 512, kernel_size=4, stride=2, padding=1, bias=self.bias, 
                            spectral_norm=self.spectral_norm, batch_norm= self.batch_norm, activation="LeakyReLU", self_attn=False)
        
        model += conv_layer(512, 1, kernel_size=(3,5), bias=self.bias, 
                            spectral_norm=False, batch_norm= False, activation=None)
        if sigmoid:
            model += torch.nn.Sigmoid()
        self.model = torch.nn.Sequential(*model)
	

    def forward(self, img, roi, value):
        #print(value.shape)
        label_embed = self.label_embedding(value)
        #print(label_embed.shape)
        label_embed = label_embed.view(-1, 1, self.img_height, self.img_width)
        D_input = torch.cat((img , roi, label_embed), dim=1)
        out = (self.model(D_input))
        return out.squeeze(2).squeeze(1)