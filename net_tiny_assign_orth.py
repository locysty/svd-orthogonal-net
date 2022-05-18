import torch
import torch.nn.functional
from torch import nn
import numpy as np
lr=0.01


#myRelu6m=nn.ReLU6()
myRelu6m=nn.LeakyReLU(negative_slope=0.3)
init_orthogonal_matrix=False
svd=True #if use svd orthogonalization(linear layer ,depthwise conv1*1)
conv_svd=False #if use svd orthogonalization in normal conv layers (3*3)
conv_one_group_no_svd=True
nNorm=True
re_assign_svd_values=False
auto_affine=True
bias_init_to_1=True #if initial batchnorm bias to 1

class myConv(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.conv=nn.Conv2d(*args,**kwargs)
        if 'groups' in kwargs:
            self.groups=kwargs['groups']
        else:
            self.groups = 1
    def forward(self,x):
        if self.training and svd and conv_svd and (not conv_one_group_no_svd or self.conv.weight.shape[0]>self.groups):
            y = self.conv.weight
            y.requires_grad=False
            out_,in_,s1,s2 =y.shape
            a=y.reshape((self.groups,-1,(in_*s1*s2)))
            a=a.transpose(1,2)
            u, s, vh = torch.linalg.svd(a, full_matrices=False)
            mat = u @ vh
            if nNorm:
                if a.shape[2] > a.shape[1]:
                    mat = mat / (torch.linalg.norm(mat, axis=0, keepdims=True).detach() + 1e-10)
            mat=mat.transpose(1,2)
            y[:] = mat.reshape((out_,in_,s1,s2))
            y.requires_grad = True
        return self.conv(x)
def myMatmul(x,y,training,axis=-1):
    if axis!=-1:
        x=x.swapaxes(axis,-1)
    if training and svd:
        y.requires_grad=False
        u, s, vh = torch.linalg.svd(y, full_matrices=False)
        mat = u @ vh
        if nNorm:
            if y.shape[1] > y.shape[0]:
                mat = mat / (torch.linalg.norm(mat, axis=0, keepdims=True).detach() + 1e-10)
        y[:] = mat
        y.requires_grad = True
    elif svd:
        u, s, vh = torch.linalg.svd(y, full_matrices=False)
        mat = u @ vh
        if nNorm:
            if y.shape[1] > y.shape[0]:
                mat = mat / (torch.linalg.norm(mat, axis=0, keepdims=True).detach() + 1e-10)
        y=mat
    x=torch.matmul(x,y)
    if axis!=-1:
        x=x.swapaxes(axis,-1)
    return x
class myPointwiseConv(torch.nn.Module):
    def __init__(self, in_features, out_features, stride=1,bias=False):
        super().__init__()
        if init_orthogonal_matrix:
            #self.weight = nn.Parameter(torch.tensor(generate_random_orthogonal_matrix(in_features,out_features),dtype=torch.float), requires_grad=True)
            self.weight = nn.Parameter(torch.empty((in_features, out_features)), requires_grad=True)
            nn.init.orthogonal_(self.weight)
        else:
            self.weight = nn.Parameter(torch.empty((in_features, out_features)), requires_grad=True)
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        self.has_bias=bias
        if bias:
            self.bias= nn.Parameter(torch.zeros((out_features,1,1)),requires_grad=True)
        self.stride=stride
    def forward(self,inputs):
        if self.stride>1:
            x=inputs[:,:,::self.stride,::self.stride]
        else:
            x=inputs
        x=myMatmul(x,self.weight,self.training,axis=1)
        if self.has_bias:x+=self.bias
        return x
class myLinar(torch.nn.Module):
    def __init__(self, in_features, out_features,bias=True):
        super().__init__()
        if init_orthogonal_matrix:
            #self.weight = nn.Parameter(torch.tensor(generate_random_orthogonal_matrix(in_features,out_features),dtype=torch.float), requires_grad=True)
            self.weight = nn.Parameter(torch.empty((in_features, out_features)), requires_grad=True)
            nn.init.orthogonal_(self.weight)
        else:
            self.weight = nn.Parameter(torch.empty((in_features,out_features)),requires_grad=True)
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        self.has_bias=bias
        if bias:
            self.bias= nn.Parameter(torch.zeros((out_features)),requires_grad=True)
    def forward(self,inputs):
        x=inputs
        x=myMatmul(x,self.weight,self.training)
        if self.has_bias:x+=self.bias
        return x
class SEModule(nn.Module):
    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            #nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            myPointwiseConv(in_, squeeze_ch,bias=True),
            Swish(),
            #nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
            myPointwiseConv(squeeze_ch, in_, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class affine(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.a=a
        self.b=b
    def forward(self,x):
        x*=self.a
        x+=self.b
        return x
#myAffine=affine(0.75,1)
myAffine=affine(1.2025,1)
#myAffine=affine(1.5,2)
#myAffine=affine(0.75,2)
#myAffine=affine(0.75,0)

#from scipy.stats import ortho_group
#m = ortho_group.rvs(dim=3)
def generate_random_orthogonal_matrix(n, m):
    while 1:
        H = np.random.randn(n, m - 1)
        ones=np.ones((n, 1), dtype=np.float)
        ones=ones* np.exp(np.log(1.5)*np.clip(np.random.randn(*ones.shape)/3,-1,1))
        ones/=np.linalg.norm(ones)
        H = np.concatenate((ones, H), axis=1) #加上几乎全等向量
        u, s, vh = np.linalg.svd(H, full_matrices=False)
        mat = u @ vh
        if m>n:
            mat/=np.linalg.norm(mat,axis=0,keepdims=True)
        a=np.linalg.norm(mat,axis=0)
        if np.all(np.logical_and(a<1.2,a>0.8)): #防止全0、nan或inf
            break
    return mat

def conv_bn(*args,**kwargs):
    if auto_affine:
        b=torch.nn.BatchNorm2d(args[1])
        if bias_init_to_1:nn.init.constant_(b.bias,1)
        return nn.Sequential(
            #torch.nn.Conv2d(bias=False, *args, **kwargs),
            myConv(bias=False, *args, **kwargs),
            b,
        )
    else:
        return nn.Sequential(
            #torch.nn.Conv2d(bias=False,*args,**kwargs),
            myConv(bias=False, *args, **kwargs),
            torch.nn.BatchNorm2d(args[1],affine=False),
            myAffine
        )
def myPointwiseConv_bn(in_features,out_features,stride=1):
    if auto_affine:
        b = torch.nn.BatchNorm2d(out_features)
        if bias_init_to_1: nn.init.constant_(b.bias, 1)
        return nn.Sequential(
            myPointwiseConv(in_features, out_features, stride=stride, bias=False),
            b,
        )
    else:
        return nn.Sequential(
            myPointwiseConv(in_features,out_features,stride=stride,bias=False),
            torch.nn.BatchNorm2d(out_features,affine=False),
            myAffine
        )
def bn_lrelu(in_features):
    if auto_affine:
        b = nn.BatchNorm2d(in_features)
        if bias_init_to_1: nn.init.constant_(b.bias, 1)
        return nn.Sequential(
            b,
            nn.LeakyReLU(negative_slope=0.3)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm2d(in_features,affine=False),
            myAffine,
            nn.LeakyReLU(negative_slope=0.3)
        )
def bn_lrelu_1d(in_features):
    if auto_affine:
        b = nn.BatchNorm1d(in_features)
        if bias_init_to_1: nn.init.constant_(b.bias, 1)
        return nn.Sequential(
            b,
            nn.LeakyReLU(negative_slope=0.3)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm1d(in_features,affine=False),
            myAffine,
            nn.LeakyReLU(negative_slope=0.3)
        )
def bn_myRelu_1d(in_features):
    if auto_affine:
        b =nn.BatchNorm1d(in_features)
        if bias_init_to_1: nn.init.constant_(b.bias, 1)
        return nn.Sequential(
            b,
            myRelu6m
        )
    else:
        return nn.Sequential(
            nn.BatchNorm1d(in_features,affine=False),
            myAffine,
            myRelu6m
        )
def conv_bn_lrelu(*args,**kwargs):
    a=conv_bn(*args,**kwargs)
    return nn.Sequential(
        a,
        nn.LeakyReLU(negative_slope=0.3)
    )
def conv_bn_myRelu6(*args,**kwargs):
    a=conv_bn(*args,**kwargs)
    return nn.Sequential(
        a,
        myRelu6m
    )
def myPointwiseConv_bn_relu6(in_features,out_features,stride=1):
    a=myPointwiseConv_bn(in_features,out_features,stride=stride)
    return nn.Sequential(
        a,
        myRelu6m,
    )
def myPointwiseConv_bn_lrelu(in_features,out_features,stride=1):
    a=myPointwiseConv_bn(in_features,out_features,stride=stride)
    return nn.Sequential(
        a,
        nn.LeakyReLU(negative_slope=0.3)
    )

class MBblock(torch.nn.Module):
    def __init__(self, n, in_features, out_features, expand=4, se_ratio=0.25, final_avti=myRelu6m, final_pool=None,
                 final_skip_con=True,final_bn=True):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.repeats=[]
        if out_features!=in_features or final_pool is not None:final_skip_con=False
        self.final_skip_con=final_skip_con
        for i in range(n):
            if i<n-1:
                self.repeats.append(
                    torch.nn.Sequential(
                        myPointwiseConv_bn_lrelu(in_features, in_features * expand),
                        conv_bn_myRelu6(in_features*expand,in_features*expand,padding='same',groups=in_features*expand,kernel_size=3),
                        #SEModule(in_features * expand,int(in_features * se_ratio)),
                        myPointwiseConv_bn_relu6(in_features*expand,in_features),
                    )
                )
            else:
                a=torch.nn.Sequential(
                    myPointwiseConv_bn_lrelu(in_features, in_features * expand),
                    conv_bn_myRelu6(in_features * expand, in_features * expand, padding='same',
                                      groups=in_features * expand, kernel_size=3),
                    #SEModule(in_features * expand, int(in_features * se_ratio)),
                )
                if final_pool is not None:
                    a=torch.nn.Sequential(a,final_pool)
                a=torch.nn.Sequential(a,
                                      myPointwiseConv_bn(in_features * expand, out_features) if final_bn else
                                      myPointwiseConv(in_features * expand,out_features,bias=False),#双线性，无偏
                                      )
                if final_avti is not None:
                    a=torch.nn.Sequential(a,final_avti)
                self.repeats.append(
                    a
                )
        self.repeats=nn.ModuleList(self.repeats)
    def forward(self,inputs):
        x=inputs
        last_x=inputs
        for I,b in enumerate(self.repeats):
            #print('dbfdsbsb ',b)
            x=b(x)
            if I<len(self.repeats)-1 or self.final_skip_con :
                x+=last_x
            last_x=x
        return x

class net(nn.Module):
    def __init__(self, out_num=10):
        super().__init__()
        self.initial= nn.Sequential(nn.ZeroPad2d(1),
                                    conv_bn_lrelu(3, 32, kernel_size=3,stride=2),
                                    )
        self.my_relu6=myRelu6m
        self.b0=MBblock(2,32,16,final_pool=nn.MaxPool2d(2, stride=2))
        self.b2 = MBblock(2, 16, 16,final_pool=nn.MaxPool2d(2, stride=2))
        self.b3 = MBblock(2, 16, 24,final_pool=nn.AdaptiveAvgPool2d(1))
        self.before_out=nn.Sequential(
                nn.Flatten(),
        )
        self.out=nn.Sequential(
                            #nn.Linear(160,120,bias=False),
                            myLinar(24,16,bias=False),
                            #bn_lrelu_1d(120),
                            bn_myRelu_1d(16),
                            #nn.Dropout(p=0.2),
                            #nn.Linear(120,out_num),
                            myLinar(16,out_num),
                            #nn.Softmax(),
                                )
    def forward(self,inputs,hook=None):
        x=inputs
        x=self.initial(x)
        if hook is not None:hook['initial']=x
        x=self.my_relu6(x)
        if hook is not None: hook['initial_relu6'] = x
        x = self.b0(x)
        if hook is not None: hook['b0'] = x
        x=self.b2(x)
        if hook is not None: hook['b2'] = x
        b2=x
        x=self.b3(x)
        if hook is not None: hook['b3'] = x
        b3=x
        x=self.before_out(x)
        if 0:
            x=self.out(x)
        else:
            for i, layer in enumerate(self.out):
                x = layer(x)
                if hook is not None and i==1:
                    hook['before_final'] = x
        return x



if torch.cuda.is_available():
    zero=torch.tensor(0, dtype=torch.float,device='cuda:0')
else:
    zero = torch.tensor(0, dtype=torch.float)

class myClip2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clip(x,-2,2)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        pos=grad_output>0
        neg=torch.logical_not(pos)
        #print(grad_output,zero,torch.logical_and(pos,x<-1))
        grad_output=torch.where(torch.logical_and(pos,x<-3),zero,grad_output)#留个margin
        grad_output = torch.where(torch.logical_and(neg, x > 3), zero, grad_output)#留个margin
        return grad_output
def work():
    the_net=net()
    the_net.eval()
    print('\n'.join(['%s   ,   %s'%(name ,_.shape) for name,_ in the_net.named_parameters()]))
    from fvcore.nn import FlopCountAnalysis
    img = torch.randn(1, 3, 32, 32)
    flops = FlopCountAnalysis(the_net, img)
    print('flops:', flops.total())
    import time
    with torch.no_grad():
        for i in range(10):
            img = torch.randn(1, 3, 32, 32)
            ti = time.time()
            pred = the_net(img)  # (1, 1000)
            print(time.time() - ti)  # 0.33

if __name__ == '__main__':
    work()
