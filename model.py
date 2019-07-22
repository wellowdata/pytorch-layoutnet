import torch
import torch.nn as nn
import torch.nn.functional as F



class LayoutNet(nn.Module):
    def __init__(self):
        super(LayoutNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc7 = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, padding=1),
            nn.ReLU(inplace=True)
        )        

        self.dec1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(1024*2, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(512*2, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(256*2, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.Conv2d(128*2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )        
        self.dec6 = nn.Sequential(
            nn.Conv2d(64*2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )        
        self.dec7 = nn.Sequential(
            nn.Conv2d(32*2, 3, kernel_size=3, padding=1),
            nn.Sigmoid()        #?
        )        

        self.jointdec2 = nn.Sequential(
            nn.Conv2d(1024*3, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.jointdec3 = nn.Sequential(
            nn.Conv2d(512*3, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.jointdec4 = nn.Sequential(
            nn.Conv2d(256*3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.jointdec5 = nn.Sequential(
            nn.Conv2d(128*3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )        
        self.jointdec6 = nn.Sequential(
            nn.Conv2d(64*3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )        
        self.jointdec7 = nn.Sequential(
            nn.Conv2d(32*3, 8, kernel_size=3, padding=1),
            nn.Sigmoid()        #?
        )        

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = F.interpolate(scale_factor=2, mode='nearest')   # if below code runs ok, try using self.upsample
        
        
    def forward(self, x):
        ## ENCODER
        encoder1 = self.enc1(x)
        encoder1_ds = self.pool(encoder1)  # ds ==> downsample
        encoder2 = self.enc2(encoder1_ds)
        encoder2_ds = self.pool(encoder2) # pool2
        encoder3 = self.enc3(encoder2_ds)
        encoder3_ds = self.pool(encoder3) # pool3
        encoder4 = self.enc4(encoder3_ds)
        encoder4_ds = self.pool(encoder4) # pool4
        encoder5 = self.enc5(encoder4_ds)
        encoder5_ds = self.pool(encoder5) # pool5
        encoder6 = self.enc6(encoder5_ds) # conv6_relu 
        encoder6_ds = self.pool(encoder6) # pool6
        encoder7 = self.enc7(encoder6_ds) # conv7_relu 
        encoder7_ds = self.pool(encoder7) # pool7


        ## DECODER
        
        unpool00 = F.interpolate(encoder7_ds,  # AKA pool7
                                    scale_factor=2, mode='nearest')
        decoder1 = torch.cat([
            self.dec1(unpool00),  #deconv00_relu
            encoder6_ds  # AKA pool6
            ]) # unpool0_
        decoder2 = torch.cat([
            self.dec2(F.interpolate(decoder1, #unpool0_
                                    scale_factor=2, mode='nearest')
                     ), #deconv0_relu 
            encoder5_ds # pool5
            ]) #unpool1_ 
        decoder3 = torch.cat([
            self.dec3(F.interpolate(decoder2, #unpool1_
                                    scale_factor=2, mode='nearest')
                     ), #deconv1_relu 
            encoder4_ds # pool4
        ]) #unpool2_ 
        decoder4 = torch.cat([
            self.dec4(F.interpolate(decoder3, #unpool2_
                                    scale_factor=2, mode='nearest')
                     ), #deconv2_relu 
            encoder3_ds # pool3
        ]) #unpool3_ 
        decoder5 = torch.cat([
            self.dec5(F.interpolate(decoder4, #unpool3_
                                    scale_factor=2, mode='nearest'
                                   )
                      ), #deconv3_relu 
            encoder2_ds # pool2
        ]) #unpool4_ 
        decoder6 = torch.cat([
            self.dec6(F.interpolate(decoder5, #unpool4_
                                    scale_factor=2, mode='nearest'
                                   )
                      ), #deconv4_relu
            encoder1_ds # pool1
        ])  #unpool5_ 
        unpool5 = F.interpolate(decoder6, scale_factor=2, mode='nearest')
        decoder7 = deconv6_sf = self.dec7(unpool5)
        
        
        ## JOINT
        
        joint1 =torch.cat([self.dec1(unpool00), #deconv00_relu_c 
                           decoder1, # unpool0_ 
                          ]) #unpool0_c
        joint2  = torch.cat([self.jointdec2(F.interpolate(joint1, scale_factor=2, mode='nearest')),
                                decoder2, #unpool1_
                               ]) #unpool1_c
        joint3  = torch.cat([self.jointdec3(F.interpolate(joint2, scale_factor=2, mode='nearest')),
                                decoder3, #unpool2_
                               ]) #unpool2_c
        joint4  = torch.cat([self.jointdec4(F.interpolate(joint3, scale_factor=2, mode='nearest')),
                                decoder4, #unpool3_
                               ]) #unpool3_c
        joint5  = torch.cat([self.jointdec5(F.interpolate(joint4, scale_factor=2, mode='nearest')),
                                decoder5, #unpool4_
                               ]) #unpool4_c
        joint6  = torch.cat([
            self.jointdec6(F.interpolate(joint4, scale_factor=2, mode='nearest')),
                                decoder6, #unpool5_
                               ]) #unpool5_c

        joint2 = self.dec2(torch.cat([F.interpolate(encoder2, 2), decoder2]))
        joint3 = self.dec3(torch.cat([F.interpolate(encoder3, 2), decoder3]))
        joint4 = self.dec4(torch.cat([F.interpolate(encoder4, 2), decoder4]))
        joint5 = self.dec5(torch.cat([F.interpolate(encoder5, 2), decoder5]))
        joint6 = sigmoid(joint5)
        
'''
unpool00 = nn.SpatialUpSamplingNearest(2)(pool7)
deconv00 = nn.SpatialConvolution(2048,1024,3,3,1,1,1,1)(unpool00)
deconv00_relu = nn.ReLU(true)(deconv00)
'''
        ## REFINEMENT

        ref0 = nn.reshape(2048 * 4 * 4) #(pool7)
        ref1 = nn.Linear(2048 * 4 * 4, 1024) #(ref0)
        ref1_relu = nn.ReLU(inplace=True) #(ref1)
        ref2 = nn.Linear(1024, 256) #(ref1_relu)
        ref2_relu = nn.ReLU(inplace=True) #(ref2)
        ref3 = nn.Linear(256, 64) #(ref2_relu)
        ref3_relu = nn.ReLU(inplace=True) #(ref3)
        roomtype = nn.Linear(64, 11) #(ref3_relu)


        return decoder6_sigmoid, decoder6_sigmoid_corners, roomtype













def conv3x3(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))


def conv3x3_down(in_planes, out_planes):
    return nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.MaxPool2d(kernel_size=2, stride=2))


class Encoder(nn.Module):
    def __init__(self, in_planes=6):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList([
            conv3x3_down(in_planes, 32),
            conv3x3_down(32, 64),
            conv3x3_down(64, 128),
            conv3x3_down(128, 256),
            conv3x3_down(256, 512),
            conv3x3_down(512, 1024),
            conv3x3_down(1024, 2048)])

    def forward(self, x):
        conv_out = []
        for conv in self.convs:
            x = conv(x)
            conv_out.append(x)
        return conv_out


class Decoder(nn.Module):
    def __init__(self, skip_num=2, out_planes=3):
        super(Decoder, self).__init__()
        self.convs = nn.ModuleList([
            conv3x3(2048, 1024),
            conv3x3(1024 * skip_num, 512),
            conv3x3(512 * skip_num, 256),
            conv3x3(256 * skip_num, 128),
            conv3x3(128 * skip_num, 64),
            conv3x3(64 * skip_num, 32)])
        self.last_conv = nn.Conv2d(
            32 * skip_num, out_planes, kernel_size=3, padding=1)

    def forward(self, f_list):
        conv_out = []
        f_last = f_list[0]
        for conv, f in zip(self.convs, f_list[1:]):
            f_last = F.interpolate(f_last, scale_factor=2, mode='nearest')
            f_last = conv(f_last)
            f_last = torch.cat([f_last, f], dim=1)
            conv_out.append(f_last)
        conv_out.append(self.last_conv(F.interpolate(
            f_last, scale_factor=2, mode='nearest')))
        return conv_out

    
class Refinement(nn.Module):
    def __init__(self, out_cat=11):  #or 8??
        super(Refinement, self).__init__()
        'torch.reshape(-1) #flatten(2048*4*4)'
        self.refine = nn.sequential([
            nn.linear(2048*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.linear(256, 64),
            nn.ReLU(inplace=True),
            nn.linear(64, out_cat),
            nn.ReLU(inplace=True),
            ])

    def forward(self, x):
        # flatten here?
        return self.refine(x)

if __name__ == '__main__':

    encoder = Encoder()
    edg_decoder = Decoder(skip_num=2, out_planes=3)
    cor_decoder = Decoder(skip_num=3, out_planes=1)

    with torch.no_grad():
        x = torch.rand(2, 6, 512, 1024)
        en_list = encoder(x)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])

    for f in en_list:
        print('encoder', f.size())
    for f in edg_de_list:
        print('edg_decoder', f.size())
    for f in cor_de_list:
        print('cor_decoder', f.size())
