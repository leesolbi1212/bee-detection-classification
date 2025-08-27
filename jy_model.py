import torch
import torch.nn as nn

def conv_bn_act(c1, c2, k=3, s=1, p=None, act=True):
    if p is None:
        p = k//2
    layers = [nn.Conv2d(c1, c2, k, s, p, bias=False),
              nn.BatchNorm2d(c2)]
    if act:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, c, hidden=None):
        super().__init__()
        h = hidden or c//2
        self.conv1 = conv_bn_act(c, h, 1,1,0)
        self.conv2 = conv_bn_act(h, c, 3,1)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        def make_stage(in_c, out_c, n_blocks):
            layers = [conv_bn_act(in_c, out_c, 3, 2)]
            for _ in range(n_blocks):
                layers.append(ResidualBlock(out_c))
            return nn.Sequential(*layers)
        self.stem = conv_bn_act(3, 32, 3,1)
        self.stage1 = make_stage(32,64,1)
        self.stage2 = make_stage(64,128,2)
        self.stage3 = make_stage(128,256,8)
        self.stage4 = make_stage(256,512,8)
        self.stage5 = make_stage(512,1024,4)
    def forward(self,x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3,c4,c5

def yolo_conv_set(c_in,c_out):
    return nn.Sequential(
        conv_bn_act(c_in,c_out,1,1,0),
        conv_bn_act(c_out,c_out*2,3,1),
        conv_bn_act(c_out*2,c_out,1,1,0),
        conv_bn_act(c_out,c_out*2,3,1),
        conv_bn_act(c_out*2,c_out,1,1,0)
    )

class YoloHead(nn.Module):
    def __init__(self,in_c,na,nc):
        super().__init__()
        self.pred = nn.Conv2d(in_c,na*(5+nc),1,1,0)
    def forward(self,x,na,nc):
        bs,_,h,w=x.shape
        p=self.pred(x)
        return p.view(bs,na,5+nc,h,w).permute(0,1,3,4,2).contiguous()

class YOLOv3(nn.Module):
    def __init__(self,nc=1,
                 anchors=((10,13),(16,30),(33,23),
                          (30,61),(62,45),(59,119),
                          (116,90),(156,198),(373,326)),
                 anchor_masks=((0,1,2),(3,4,5),(6,7,8))):
        super().__init__()
        self.nc=nc
        self.anchors=torch.tensor(anchors,dtype=torch.float32)
        self.anchor_masks=anchor_masks
        self.na=3
        self.strides=(8,16,32)
        self.backbone=Darknet53()
        self.head_l_conv=yolo_conv_set(1024,512)
        self.head_l_out=YoloHead(512,self.na,nc)
        self.reduce_l_to_m=conv_bn_act(512,256,1,1,0)
        self.head_m_conv=yolo_conv_set(256+512,256)
        self.head_m_out=YoloHead(256,self.na,nc)
        self.reduce_m_to_s=conv_bn_act(256,128,1,1,0)
        self.head_s_conv=yolo_conv_set(128+256,128)
        self.head_s_out=YoloHead(128,self.na,nc)
        self.upsample=nn.Upsample(scale_factor=2,mode='nearest')
    def forward(self,x):
        c3,c4,c5=self.backbone(x)
        x_l=self.head_l_conv(c5)
        p_l=self.head_l_out(x_l,self.na,self.nc)
        x_m=self.reduce_l_to_m(x_l)
        x_m=self.upsample(x_m)
        x_m=torch.cat([x_m,c4],1)
        x_m=self.head_m_conv(x_m)
        p_m=self.head_m_out(x_m,self.na,self.nc)
        x_s=self.reduce_m_to_s(x_m)
        x_s=self.upsample(x_s)
        x_s=torch.cat([x_s,c3],1)
        x_s=self.head_s_conv(x_s)
        p_s=self.head_s_out(x_s,self.na,self.nc)
        return [p_s,p_m,p_l]
    @staticmethod
    def decode(preds,anchors,anchor_masks,strides,img_size):
        outs=[]
        device=preds[0].device
        for i,p in enumerate(preds):
            bs,na,gh,gw,no=p.shape
            stride=strides[i]
            a_idx=anchor_masks[i]
            a=anchors[torch.tensor(a_idx,device=device),:]
            yv,xv=torch.meshgrid(torch.arange(gh,device=device),torch.arange(gw,device=device),indexing='ij')
            grid=torch.stack((xv,yv),2).view(1,1,gh,gw,2).float()
            px=p[...,:2].sigmoid()
            pw=p[...,2:4]
            obj=p[...,4:5].sigmoid()
            cls=p[...,5:].sigmoid()
            xy=(px+grid)*stride
            wh=pw.exp()*a.view(1,na,1,1,2)
            box=torch.cat([xy,wh],-1).view(bs,-1,4)
            score=(obj*cls.max(-1,keepdim=True).values).view(bs,-1,1)
            cls_id=cls.argmax(-1).view(bs,-1,1).float()
            outs.append(torch.cat([box,score,cls_id],-1))
        return torch.cat(outs,1)

def yolo_v3(nc=1):
    return YOLOv3(nc=nc)
