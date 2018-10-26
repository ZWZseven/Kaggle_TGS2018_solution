from model import *

class Double_unit(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size=3, padding=1):#in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size,C,H,W = x.shape
        return self.layer(x)
class Standard_unit(nn.Module):
    def __init__(self, stage, nb_filter, kernel_size=3, padding=1):#in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(nb_filter, nb_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter, nb_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size,C,H,W = x.shape
        return self.layer(x)

class Standard_unit1(nn.Module):
    def __init__(self, stage, in_c, nb_filter, kernel_size=3, padding=1):#in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(in_c, nb_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter, nb_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size,C,H,W = x.shape
        return self.layer(x)
    
class Ups(nn.Module):
    def __init__(self, in_chan, out_chan, upscale=1):#in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        if upscale == 1:#Upscale.transposed_conv:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_chan, out_chan, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True)
            )
        elif upscale == 2:#Upscale.upsample_bilinear:
            self.layer = nn.Upsample(scale_factor=2,mode='bilinear')
            nn.Conv2d(in_chan, out_chan, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        else:
            self.layer = nn.PixelShuffle(upscale_factor=2)


    def forward(self, x):
        batch_size,C,H,W = x.shape
        return self.layer(x)    

"""
def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x
"""
"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""
class UnetPP(nn.Module):
    def __init__(self, num_class):#in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        nb_filter = [32,64,128,256,512]
        
        self.conv1_1 = Standard_unit1(stage='11',in_c=3, nb_filter=nb_filter[0])
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = Standard_unit1(stage='21',in_c=nb_filter[0], nb_filter=nb_filter[1])
        self.pool2 = nn.MaxPool2d(2, 2)

        self.up1_2 = Ups(nb_filter[1],nb_filter[0])#Conv2DTranspose(nb_filter[0])(conv2_1)
        #self.conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
        self.conv1_2 = Standard_unit1(stage='12',in_c=nb_filter[0]+nb_filter[0], nb_filter=nb_filter[0])

        self.conv3_1 = Standard_unit1( stage='31', in_c=nb_filter[1],nb_filter=nb_filter[2])
        self.pool3 = nn.MaxPool2d(2, 2)

        self.up2_2 = Ups(nb_filter[2],nb_filter[1])#Conv2DTranspose(nb_filter[1])(conv3_1)
        #self.conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
        self.conv2_2 = Standard_unit1( stage='22', in_c=nb_filter[1]+nb_filter[1], nb_filter=nb_filter[1])

        self.up1_3 = Ups(nb_filter[1],nb_filter[0])#Conv2DTranspose(nb_filter[0])(conv2_2)
        #self.conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
        self.conv1_3 = Standard_unit1( stage='13', in_c=nb_filter[0]*3, nb_filter=nb_filter[0])

        self.conv4_1 = Standard_unit1(stage='41',in_c=nb_filter[2], nb_filter=nb_filter[3])
        self.pool4 = nn.MaxPool2d(2, 2)

        self.up3_2 = Ups(nb_filter[3],nb_filter[2])#Conv2DTranspose(nb_filter[2])(conv4_1)
        #self.conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
        self.conv3_2 = Standard_unit1(stage='32',in_c=nb_filter[2]+nb_filter[2], nb_filter=nb_filter[2])

        self.up2_3 = Ups(nb_filter[2],nb_filter[1])#Conv2DTranspose(nb_filter[1])(conv3_2)
        #self.conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
        self.conv2_3 = Standard_unit1( stage='23', in_c=nb_filter[1]*3,nb_filter=nb_filter[1])

        self.up1_4 = Ups(nb_filter[1],nb_filter[0])#Conv2DTranspose(nb_filter[0])(conv2_3)
        #self.conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
        self.conv1_4 = Standard_unit1(stage='14', in_c=nb_filter[0]*4,nb_filter=nb_filter[0])

        self.conv5_1 = Standard_unit1( stage='51',in_c=nb_filter[3], nb_filter=nb_filter[4])

        self.up4_2 = Ups(nb_filter[4],nb_filter[3])#Conv2DTranspose(nb_filter[3])(conv5_1)
        #self.conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
        self.conv4_2 = Standard_unit1(stage='42', in_c=nb_filter[3]+nb_filter[3],nb_filter=nb_filter[3])

        self.up3_3 = Ups(nb_filter[3],nb_filter[2])#Conv2DTranspose(nb_filter[2])(conv4_2)
        #self.conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
        self.conv3_3 = Standard_unit1( stage='33', in_c=nb_filter[2]*3,nb_filter=nb_filter[2])

        self.up2_4 = Ups(nb_filter[2],nb_filter[1])#Conv2DTranspose(nb_filter[1])(conv3_3)
        #self.conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
        self.conv2_4 = Standard_unit1(stage='24',in_c=nb_filter[1]*4, nb_filter=nb_filter[1])

        self.up1_5 = Ups(nb_filter[1],nb_filter[0])#Conv2DTranspose(nb_filter[0])(conv2_4)
        #self.conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
        self.conv1_5 = Standard_unit1(stage='15', in_c=nb_filter[0]*5,nb_filter=nb_filter[0])        
        
        self.nestnet_output_1 = nn.Conv2d(nb_filter[0],num_class, 1, padding=0)#(conv1_2)
        self.nestnet_output_2 = nn.Conv2d(nb_filter[0],num_class, 1, padding=0)#(conv1_3)
        self.nestnet_output_3 = nn.Conv2d(nb_filter[0],num_class, 1, padding=0)#(conv1_4)
        self.nestnet_output_4 = nn.Conv2d(nb_filter[0],num_class, 1, padding=0)#(conv1_5)        
        
        #= nn.Conv2d(num_class,num_class, 1, padding=0)#
    def forward(self, img_input):#, img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
        img_input=img_input.float()
        #nb_filter = [32,64,128,256,512]

        # Handle Dimension Ordering for different backends
        #global bn_axis
        #if K.image_dim_ordering() == 'tf':
        #  bn_axis = 3
        #  img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
        #else:
        #  bn_axis = 1
        #  img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')
        bn_axis = 1
        
        conv1_1 = self.conv1_1(img_input)
        pool1 = self.pool1(conv1_1)

        conv2_1 = self.conv2_1(pool1)
        pool2 = self.pool2(conv2_1)

        up1_2 = self.up1_2(conv2_1)
        conv1_2 = torch.cat([up1_2, conv1_1],  bn_axis)
        conv1_2 = self.conv1_2(conv1_2)

        conv3_1 = self.conv3_1(pool2)
        pool3 = self.pool3(conv3_1)

        up2_2 = self.up2_2(conv3_1)
        conv2_2 = torch.cat([up2_2, conv2_1],  bn_axis)
        conv2_2 = self.conv2_2(conv2_2)

        up1_3 = self.up1_3(conv2_2)
        conv1_3 = torch.cat([up1_3, conv1_1, conv1_2],  bn_axis)
        conv1_3 = self.conv1_3(conv1_3)

        conv4_1 = self.conv4_1(pool3)
        pool4 = self.pool4(conv4_1)

        up3_2 = self.up3_2(conv4_1)
        conv3_2 = torch.cat([up3_2, conv3_1],  bn_axis)
        conv3_2 = self.conv3_2(conv3_2)

        up2_3 = self.up2_3(conv3_2)
        conv2_3 = torch.cat([up2_3, conv2_1, conv2_2],  bn_axis)
        conv2_3 = self.conv2_3(conv2_3)

        up1_4 = self.up1_4(conv2_3)
        conv1_4 = torch.cat([up1_4, conv1_1, conv1_2, conv1_3], bn_axis)
        conv1_4 = self.conv1_4(conv1_4)

        conv5_1 = self.conv5_1(pool4)

        up4_2 = self.up4_2(conv5_1)
        conv4_2 = torch.cat([up4_2, conv4_1],  bn_axis)
        conv4_2 = self.conv4_2(conv4_2)

        up3_3 = self.up3_3(conv4_2)
        conv3_3 = torch.cat([up3_3, conv3_1, conv3_2],  bn_axis)
        conv3_3 = self.conv3_3(conv3_3)

        up2_4 = self.up2_4(conv3_3)
        conv2_4 = torch.cat([up2_4, conv2_1, conv2_2, conv2_3],  bn_axis)
        conv2_4 = self.conv2_4(conv2_4)

        up1_5 = self.up1_5(conv2_4)
        conv1_5 = torch.cat([up1_5, conv1_1, conv1_2, conv1_3, conv1_4],  bn_axis)
        conv1_5 = self.conv1_5(conv1_5)
    
        nestnet_output_1 = self.nestnet_output_1(conv1_2)
        nestnet_output_2 = self.nestnet_output_2(conv1_3)
        nestnet_output_3 = self.nestnet_output_3(conv1_4)
        nestnet_output_4 = self.nestnet_output_4(conv1_5)
        """
        if deep_supervision:
            model = Model(input=img_input, output=[nestnet_output_1,
                                                   nestnet_output_2,
                                                   nestnet_output_3,
                                                   nestnet_output_4])
        else:
            model = Model(input=img_input, output=[nestnet_output_4])
        """
        return nestnet_output_4, nestnet_output_3, nestnet_output_2, nestnet_output_1

    
    
    
    
class LinkNet34deeps(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super().__init__()
        assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.conv4=nn.Sequential(
            nn.Conv2d(512,256,1),
            nn.BatchNorm2d(256),
            nonlinearity(inplace=True),
        ) #nn.Conv2d(512,256,1) 
        self.conv44f=nn.Conv2d(64,1,1)
        self.conv44=nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nonlinearity(inplace=True),
            #nn.Conv2d(64, 1, 1),
        ) 
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.conv3=nn.Sequential(
            nn.Conv2d(256,128,1),
            nn.BatchNorm2d(128),
            nonlinearity(inplace=True),
        ) #nn.Conv2d(256,128,1) 
        self.conv33f=nn.Conv2d(64,1,1)
        self.conv33= nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nonlinearity(inplace=True),
            #nn.Conv2d(64, 1, 1),
        ) 
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.conv2 = nn.Sequential(
            nn.Conv2d(128,64,1),
            nn.BatchNorm2d(64),
            nonlinearity(inplace=True),
        ) #nn.Conv2d(128,64,1)  
        self.conv22f=nn.Conv2d(64,1,1)
        self.conv22=nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nonlinearity(inplace=True),
            #nn.Conv2d(64, 1, 1),
        ) 
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = ConvUp(filters[0], filters[0])
        
        # Final Classifier
        self.logit = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nonlinearity(inplace=True),
            nn.Conv2d(64, 1, 1),
        )        
        self.logit_image = nn.Sequential(
            #nn.Linear(512, 128),
            #nn.ReLU(inplace=True),
            #nn.Linear(128, 1),
            nn.Linear(64, 1),
        )
        self.fuse_image = nn.Sequential(
            nn.Linear(512, 64),
            #nn.ReLU(inplace=True),
            #nn.Linear(128, 1),
        )

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        batch_size,dim,m,n=x.shape
        x = x.float()
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        #x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = torch.cat([self.decoder4(e4) , e3], 1)#concat([self.decoder5(e5) , e4])
        d4 = self.conv4(d4)
        f4 = self.conv44(d4)
        # d4 = e3
        #d3 = self.decoder3(d4) + e2
        #print(e2.shape)
        d3 = torch.cat([self.decoder3(d4) , e2], 1)#concat([self.decoder5(e5) , e4])
        #print(d3.shape)
        d3 = self.conv3(d3)
        f3 = self.conv33(d3)
        
        #d2 = self.decoder2(d3) + e1
        d2 = torch.cat([self.decoder2(d3) , e1], 1)#concat([self.decoder5(e5) , e4])
        d2 = self.conv2(d2)
        f2 = self.conv22(d2)
        
        #d1 = self.decoder1(d2)

        # Final Classification
        f = self.finaldeconv1(d2)
        #f = self.finalrelu1(f)
        #f = self.logit(f)
          
        f21 = F.upsample(f2,scale_factor= 2, mode='bilinear',align_corners=False)
        f22 = self.conv22f(f21)  
        
        f31 = F.upsample(f3,scale_factor= 4, mode='bilinear',align_corners=False)
        f33 = self.conv33f(f31)  
        
        f41 = F.upsample(f4,scale_factor= 8, mode='bilinear',align_corners=False)
        f44 = self.conv44f(f41)  

        e = F.adaptive_avg_pool2d(e4, output_size=1).view(batch_size,-1) #image pool#-512-1-1
        e = F.dropout(e, p=0.50, training=self.training)#
        fuse_image  = self.fuse_image(e)#-64-1-1
        logit_image = self.logit_image(fuse_image).view(-1)#-1-1-1

        fuse_pixel=f+f21+f31+f41
        
        fuse = torch.cat([ #fuse
            fuse_pixel,
            F.upsample(fuse_image.view(batch_size,-1,1,1,),scale_factor=192, mode='nearest')
        ],1)     
        
               
        f = self.logit(fuse)   
        
        return f, f22, f33, f44, logit_image
    
        """
        self.fuse_pixel  = nn.Sequential(        
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.logit_pixel  = nn.Sequential(
            #nn.Conv2d(320, 64, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.Conv2d( 64,  1, kernel_size=1, padding=0),
        )

        self.logit_image = nn.Sequential(
            #nn.Linear(512, 128),
            #nn.ReLU(inplace=True),
            #nn.Linear(128, 1),
            nn.Linear(64, 1),
        )
        self.fuse_image = nn.Sequential(
            nn.Linear(512, 64),
            #nn.ReLU(inplace=True),
            #nn.Linear(128, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.logit = nn.Sequential(
            #nn.Conv2d(128, 64, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.Conv2d( 64,  1, kernel_size=1, padding=0),
        )    
        """
    
    
    
def criterion(output_4, output_3, output_2, output_1, logit_image, truth_pixel, is_average=True):
    #print(torch.sum(torch.sum(truth_pixel)))
    bat,dim,m,n=truth_pixel.shape

    truth_image = torch.Tensor(bat)
    i=0
    for x in truth_pixel:
        truth_image[i]=(x.squeeze().sum()>0).cuda().float()
        i=i+1

    truth_image=truth_image.type(torch.cuda.FloatTensor)

    loss_image = F.binary_cross_entropy_with_logits(logit_image, truth_image, reduce=is_average)#1-1-1
    
    loss_logit1 = F.binary_cross_entropy_with_logits(output_1, truth_pixel)#1-1-1
    loss_logit2 = F.binary_cross_entropy_with_logits(output_2, truth_pixel)#1-1-1
    loss_logit3 = F.binary_cross_entropy_with_logits(output_3, truth_pixel)#1-1-1
    loss_logit4 = F.binary_cross_entropy_with_logits(output_4, truth_pixel)#1-1-1

    #weight_image, weight_pixel = 0.1, 10  #focal
    weight_1, weight_2, weight_3, weight_4, weight_image = 0.2, 0.2, 0.2, 1, 0.05  #lovasz?
    #weight_image, weight_pixel = 0.1, 2 #bce



    return weight_1*loss_logit1+ weight_2*loss_logit2+weight_3*loss_logit3+weight_4*loss_logit4+weight_image*loss_image



################################################################

def sum_parameter(model):
    #print(model)
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))



