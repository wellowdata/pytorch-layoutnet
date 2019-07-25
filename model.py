import torch
import torch.nn as nn
import torch.nn.functional as F


class LayoutNet(nn.Module):
    def __init__(self):
        super(LayoutNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1)
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, stride=1)
        self.deconv00 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1, stride=1)
        self.deconv0 = nn.Conv2d(1024*2, 512, kernel_size=3, padding=1, stride=1)
        self.deconv1 = nn.Conv2d(512*2, 256, kernel_size=3, padding=1, stride=1)
        self.deconv2 = nn.Conv2d(256*2, 128, kernel_size=3, padding=1, stride=1)
        self.deconv3 = nn.Conv2d(128*2, 64, kernel_size=3, padding=1, stride=1)
        self.deconv4 = nn.Conv2d(64*2, 32, kernel_size=3, padding=1, stride=1)
        self.deconv5 = nn.Conv2d(32*2, 3, kernel_size=3, padding=1, stride=1)
        self.deconv6_sf = nn.Sigmoid()
        self.deconv00_c = nn.Conv2d(2048, 1024, kernel_size=3, padding=1, stride=1)
        self.deconv0_c = nn.Conv2d(1024*3, 512, kernel_size=3, padding=1, stride=1)
        self.deconv1_c = nn.Conv2d(512*3, 256, kernel_size=3, padding=1, stride=1)
        self.deconv2_c = nn.Conv2d(256*3, 128, kernel_size=3, padding=1, stride=1)
        self.deconv3_c = nn.Conv2d(128*3, 64, kernel_size=3, padding=1, stride=1)
        self.deconv4_c = nn.Conv2d(64*3, 32, kernel_size=3, padding=1, stride=1)
        self.deconv5_c = nn.Conv2d(32*3, 16, kernel_size=3, padding=1, stride=1)
        self.deconv6_sf_c = nn.Sigmoid()
        self.ref1 = nn.Linear(2048*4*4, 1024)
        self.ref2 = nn.Linear(1024, 256)
        self.ref3 = nn.Linear(256, 64)
        self.ref4 = nn.Linear(64, 11)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):

        ## ENCODER
        conv1 = self.conv1(x)
        conv1_relu = self.relu(conv1)
        pool1 = self.pool(conv1_relu)
        conv2 = self.conv2(pool1)
        conv2_relu = self.relu(conv2)
        pool2 = self.pool(conv2_relu)
        conv3 = self.conv3(pool2)
        conv3_relu = self.relu(conv3)
        pool3 = self.pool(conv3_relu)
        conv4 = self.conv4(pool3)
        conv4_relu = self.relu(conv4)
        pool4 = self.pool(conv4_relu)
        conv5 = self.conv5(pool4)
        conv5_relu = self.relu(conv5)
        pool5 = self.pool(conv5_relu)
        conv6 = self.conv6(pool5)
        conv6_relu = self.relu(conv6)
        pool6 = self.pool(conv6_relu)
        conv7 = self.conv7(pool6)
        conv7_relu = self.relu(conv7)
        pool7 = self.pool(conv7_relu)

        ## DECODER
        unpool00 = F.interpolate(pool7, scale_factor=2)
        deconv00 = self.deconv00(unpool00)
        deconv00_relu = self.relu(deconv00)
        unpool0_ = torch.cat((deconv00_relu, pool6), dim=1)
        unpool0 = F.interpolate(unpool0_, scale_factor=2)
        deconv0 = self.deconv0(unpool0)
        deconv0_relu = self.relu(deconv0)
        unpool1_ = torch.cat((deconv0_relu, pool5), dim=1)
        unpool1 = F.interpolate(unpool1_, scale_factor=2)
        deconv1 = self.deconv1(unpool1)
        deconv1_relu = self.relu(deconv1)
        unpool2_ = torch.cat((deconv1_relu, pool4), dim=1)
        unpool2 = F.interpolate(unpool2_, scale_factor=2)
        deconv2 = self.deconv2(unpool2)
        deconv2_relu = self.relu(deconv2)
        unpool3_ = torch.cat((deconv2_relu, pool3), dim=1)
        unpool3 = F.interpolate(unpool3_, scale_factor=2)
        deconv3 = self.deconv3(unpool3)
        deconv3_relu = self.relu(deconv3)
        unpool4_ = torch.cat((deconv3_relu, pool2), dim=1)
        unpool4 = F.interpolate(unpool4_, scale_factor=2)
        deconv4 = self.deconv4(unpool4)
        deconv4_relu = self.relu(deconv4)
        unpool5_ = torch.cat((deconv4_relu, pool1), dim=1)
        unpool5 = F.interpolate(unpool5_, scale_factor=2)
        deconv5 = self.deconv5(unpool5)
        deconv6_sf = self.deconv6_sf(deconv5)

        ## JOINT
        deconv00_c = self.deconv00_c(unpool00)
        deconv00_relu_c = self.relu(deconv00_c)
        unpool0_c = torch.cat((deconv00_relu_c, unpool0_), dim=1)
        unpool0_c = F.interpolate(unpool0_c, scale_factor=2)
        deconv0_c = self.deconv0_c(unpool0_c)
        deconv0_relu_c = self.relu(deconv0_c)
        unpool1_c = torch.cat((deconv0_relu_c, unpool1_), dim=1)
        unpool1_c = F.interpolate(unpool1_c, scale_factor=2)
        deconv1_c = self.deconv1_c(unpool1_c)
        deconv1_relu_c = self.relu(deconv1_c)
        unpool2_c = torch.cat((deconv1_relu_c, unpool2_), dim=1)
        unpool2_c = F.interpolate(unpool2_c, scale_factor=2)
        deconv2_c = self.deconv2_c(unpool2_c)
        deconv2_relu_c = self.relu(deconv2_c)
        unpool3_c = torch.cat((deconv2_relu_c, unpool3_), dim=1)
        unpool3_c = F.interpolate(unpool3_c, scale_factor=2)
        deconv3_c = self.deconv3_c(unpool3_c)
        deconv3_relu_c = self.relu(deconv3_c)
        unpool4_c = torch.cat((deconv3_relu_c, unpool4_), dim=1)
        unpool4_c = F.interpolate(unpool4_c, scale_factor=2)
        deconv4_c = self.deconv4_c(unpool4_c)
        deconv4_relu_c = self.relu(deconv4_c)
        unpool5_c = torch.cat((deconv4_relu_c, unpool5_), dim=1)
        unpool5_c = F.interpolate(unpool5_c, scale_factor=2)
        deconv5_c = self.deconv5_c(unpool5_c)
        deconv6_sf_c = self.deconv6_sf_c(deconv5_c)

        ## REFINEMENT
        ref0 = pool7.view(-1, 2048*4*4)
        ref1 = self.ref1(ref0)
        ref1_relu = self.relu(ref1)
        ref2 = self.ref2(ref1_relu)
        ref2_relu = self.relu(ref2)
        ref3 = self.ref3(ref2_relu)
        ref3_relu = self.relu(ref3)
        ref4 = self.ref4(ref3_relu)

        return deconv6_sf, deconv6_sf_c, ref4


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = LayoutNet()
    num_params = count_parameters(model)
    print(num_params - 128187606)
    x = torch.rand(2, 3, 512, 512)
    #out = model(x)








































"""
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc7 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
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
        #self.upsample = F.interpolate(scale_factor=2, mode='nearest')   # if below code runs ok, try using self.upsample
        self.relu = nn.ReLU(inplace=True)

        #ref0 = torch.reshape(2048 * 4 * 4)(encoder7_ds) #(pool7)
        self.ref1 = nn.Linear(2048 * 4 * 4, 1024)
        self.ref2 = nn.Linear(1024, 256)
        self.ref3 = nn.Linear(256, 64)
        self.roomtype = nn.Linear(64, 11)
        
    def forward(self, x):
        print("x: ", x.shape)
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
        print('pool6 ', encoder6_ds.shape)
        encoder7 = self.enc7(encoder6_ds) # conv7_relu 
        encoder7_ds = self.pool(encoder7) # pool7
        print('pool7 shape: ', encoder7_ds.shape)


        ## DECODER
        
        unpool00 = F.interpolate(encoder7_ds,  # AKA pool7
                                    scale_factor=2, mode='nearest')
        #print(unpool00.shape)
        deconv00_relu = self.dec1(unpool00)
        decoder1 = torch.cat([
            deconv00_relu,  #deconv00_relu
            encoder6_ds  # AKA pool6
            ],  dim=1) # unpool0_
        print('deconv00_relu ', self.dec1(unpool00).shape)
        print('decoder1 shape: ', decoder1.shape)
        print("upsample: ", F.interpolate(decoder1, #unpool0_
                                    scale_factor=2, mode='nearest').shape)
        deconv0_relu = self.dec2(F.interpolate(decoder1, #unpool0_
                                    scale_factor=2, mode='nearest')
                     )
        decoder2 = torch.cat([
            deconv0_relu, #deconv0_relu 
            encoder5_ds # pool5
            ],  dim=1) #unpool1_ 
        decoder3 = torch.cat([
            self.dec3(F.interpolate(decoder2, #unpool1_
                                    scale_factor=2, mode='nearest')
                     ), #deconv1_relu 
            encoder4_ds # pool4
        ],  dim=1) #unpool2_ 
        decoder4 = torch.cat([
            self.dec4(F.interpolate(decoder3, #unpool2_
                                    scale_factor=2, mode='nearest')
                     ), #deconv2_relu 
            encoder3_ds # pool3
        ],  dim=1) #unpool3_ 
        decoder5 = torch.cat([
            self.dec5(F.interpolate(decoder4, #unpool3_
                                    scale_factor=2, mode='nearest'
                                   )
                      ), #deconv3_relu 
            encoder2_ds # pool2
        ],  dim=1) #unpool4_ 
        decoder6 = torch.cat([
            self.dec6(F.interpolate(decoder5, #unpool4_
                                    scale_factor=2, mode='nearest'
                                   )
                      ), #deconv4_relu
            encoder1_ds # pool1
        ],  dim=1)  #unpool5_ 
        unpool5 = F.interpolate(decoder6, scale_factor=2, mode='nearest')
        decoder6_sigmoid = self.dec7(unpool5)
        
        
        ## JOINT
        
        joint1 =torch.cat([
            self.dec1(unpool00), #deconv00_relu_c 
                           decoder1, # unpool0_ 
                          ],  dim=1) #unpool0_c
        print(joint1.shape)
        joint2  = torch.cat([
            self.jointdec2(F.interpolate(joint1, scale_factor=2, mode='nearest')),
                                decoder2, #unpool1_
                               ],  dim=1) #unpool1_c
        joint3  = torch.cat([
            self.jointdec3(F.interpolate(joint2, scale_factor=2, mode='nearest')),
                                decoder3, #unpool2_
                               ],  dim=1) #unpool2_c
        joint4  = torch.cat([
            self.jointdec4(F.interpolate(joint3, scale_factor=2, mode='nearest')),
                                decoder4, #unpool3_
                               ],  dim=1) #unpool3_c
        joint5  = torch.cat([
            self.jointdec5(F.interpolate(joint4, scale_factor=2, mode='nearest')),
                                decoder5, #unpool4_
                               ],  dim=1) #unpool4_c
        joint6  = torch.cat([
            self.jointdec6(F.interpolate(joint5, scale_factor=2, mode='nearest')),
                                decoder6, #unpool5_
                               ],  dim=1) #unpool5_c
        
        
        decoder6_sigmoid_corners = self.jointdec7(F.interpolate(joint6, scale_factor=2, mode='nearest'))
        

        ## REFINEMENT

#        ref0 = encoder7_ds.flatten() #torch.reshape(2048 * 4 * 4)(encoder7_ds) #(pool7)
        ref1 = self.ref1(encoder7_ds.view(-1, 2048 * 4 * 4))
        print('ref1 ', ref1.shape)
        ref1_relu = self.relu(ref1)
        ref2 = self.ref2(ref1_relu)
        ref2_relu = self.relu(ref2)
        ref3 = self.ref3(ref2_relu)
        ref3_relu = self.relu(ref3)
        roomtype = self.roomtype(ref3_relu)


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
            #print('f_last shape:', f_last.shape)
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
    my_layoutnet = LayoutNet()

    with torch.no_grad():
        x_orig = torch.rand(2, 6, 512, 1024)
        x = torch.rand(2, 3, 512, 512)
        en_list = encoder(x_orig)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])
        my_layoutnet_outputs = [decoder6_sigmoid, decoder6_sigmoid_corners, roomtype
                               ] = my_layoutnet(x)
    '''
    for f in en_list:
        print('encoder', f.size())
    for f in edg_de_list:
        print('edg_decoder', f.size())
    for f in cor_de_list:
        print('cor_decoder', f.size())
    '''
    for f in my_layoutnet_outputs:
        print('my_layoutnet', f.size())

"""

