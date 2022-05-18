seed=2022
#seed=None
import torch
import random
import numpy as np
if seed is not None :
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
from torch import nn
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 64
batch_size_group=64

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#import net_my2 as net
import net_tiny_assign_orth as net
#import net_tiny_assign_orth2 as net
#import net_tiny as net
import torch.utils.tensorboard
import os,time,glob
import util
prefix='_Tiny_BiasTo1_noSVD'
#prefix='_Tiny_BiasTo1_SVD_nNorm_AssignOrth_AllOrth'
#prefix='_Tiny_BiasNotTo1_SVD_nNorm_AssignOrth'
#prefix='_Tiny_BiasTo1_SVD_nNorm_AssignOrth'

#prefix='_Tiny_BiasTo1_SVD_nNorm'
#prefix='_Tiny_BiasTo1_SVD_nNorm_from_model-acc0.748900000-loss0.491045-orthloss0.029047-step1344'
#prefix='_Tiny_BiasTo1_SVD_nNorm_from_model-acc0.747300000-loss0.541896-orthloss0.027877-step761'
#prefix='_Tiny_BiasTo1_noSVD_from_model-acc0.748100000-loss0.501927-orthloss0.032914-step1344'
#prefix='_Tiny_BiasTo1_noSVD_from_model-acc0.744000000-loss0.559740-orthloss0.033012-step925'

record_save_path='record_Tiny_BiasTo1_noSVD.npy'
reload=False
reload_step=False
do_record=True
save_record=False
save_all_show=True
feature_map_show_num=3
record_test_size=200
channel_thres=0.1
weight_margin=0.7
show_scalars=10
show_lag_div_test_time=5
get_grad_hessian=False

writer = torch.utils.tensorboard.SummaryWriter(log_dir='./summary'+prefix,max_queue=2)
the_net=net.net()
def change_momentum(momentum):
    for layer in the_net.modules():
        if isinstance(layer,nn.BatchNorm1d) or isinstance(layer,nn.BatchNorm2d) or isinstance(layer,nn.BatchNorm3d):
            layer.momentum=momentum

save_dir=os.path.join('.','model'+prefix)
save_path=os.path.join(save_dir,'model.pth')
load_path=save_path

os.makedirs(save_dir,exist_ok=True)
if reload and os.path.exists(load_path):
    the_net.load_state_dict(torch.load(load_path))
    print('loaded..')
if torch.cuda.is_available():
    the_net.cuda()
step_file='steps%s.txt'%prefix
if reload and reload_step and os.path.exists(step_file):
    with open(step_file,'r',encoding='utf8') as f:
        step=int(f.read())
else:
    step=0
lrrr=batch_size_group/600
change_momentum(momentum=0.1)
warm_up_lr=1*lrrr
warm_up_end_momentum=0.1
net.lr=warm_up_lr
#warm_up_optimizer = torch.optim.AdamW(the_net.parameters(),warm_up_lr, weight_decay=1e-4)
warm_up_optimizer = torch.optim.SGD(the_net.parameters(),warm_up_lr,momentum=0.9, weight_decay=1e-4)
#lrr=1
lrr=200*lrrr
lr=0.005*lrr
epochs=30
#optimizer = torch.optim.AdamW(the_net.parameters(),lr, weight_decay=1e-4)
#optimizer = torch.optim.AdamW(the_net.parameters(),lr,betas=(0.,0.), weight_decay=1e-4,)
optimizer=torch.optim.SGD(the_net.parameters(),lr,momentum=0.9, weight_decay=1e-4)
def change_lr_(optim,lr):
    for g in optim.param_groups:
        g['lr'] = lr
change_lr=True
change_lr_epoch_rate1=0.35
change_lr1=0.0005*lrr
change_momentum1=0.05
change_lr_epoch_rate2=0.8
change_lr2=0.00008*lrr
change_momentum2=0.05
use_cos=False
if use_cos:
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)
    #print(scheduler1.get_last_lr(),type(scheduler1.get_last_lr()))
criterion = torch.nn.CrossEntropyLoss()
save_lag = 60
last_save_time = time.time()
show_lag=75
last_show_time=0
def best_acc_saved(floats=4):
    files=glob.glob(save_dir+os.path.sep+'model-acc*.pth')
    def get_acc_str(path):
        path=os.path.split(path)[1]
        return path[9:9+floats+2]
    if files:
        max=get_acc_str(files[0])
        #print('max',max)
        for path in files[1:]:
            s=get_acc_str(path)
            if s>max:max=s
        return max
    else:
        return None
record_params=dict(
conv_weight={'initial.1.0.0.conv.weight':None,
                    'b0.repeats.0.1.0.0.conv.weight':None,
                    'b2.repeats.0.1.0.0.conv.weight':None,
                'b2.repeats.1.0.0.0.1.0.0.conv.weight':None,
                'b3.repeats.0.1.0.0.conv.weight': None,
                    'b3.repeats.1.0.0.0.1.0.0.conv.weight':None,

                    },
bn_bias={'initial.1.0.1.bias':None,
             'b0.repeats.0.0.0.1.bias':None,
            'b0.repeats.1.0.0.0.0.0.1.bias':None,
            'b2.repeats.0.0.0.1.bias':None,
                'b2.repeats.1.0.0.0.0.0.1.bias':None,
            'b3.repeats.0.0.0.1.bias':None,
                'b3.repeats.1.0.0.0.1.0.1.bias':None,
            'out.1.0.bias':None,
            'out.2.bias':None,
},
linear_weight={
    'b0.repeats.0.0.0.0.weight':None,
    'b0.repeats.1.0.1.0.weight':None,
    'b2.repeats.0.0.0.0.weight':None,
    'b2.repeats.1.0.1.0.weight':None,
    'b3.repeats.0.0.0.0.weight':None,
    'b3.repeats.0.2.0.0.weight':None,
    'out.0.weight':None,
    'out.2.weight':None,
}
)
record_params['bn_weight']={key[:-4]+'weight':None for key in record_params['bn_bias'].keys() if key !='out.2.bias'}

for name,param in the_net.named_parameters():
    for key in record_params.keys():
        for param_name in record_params[key]:
            if param_name==name:
                record_params[key][param_name]=param
                break
        else:
            continue
        break
for k1,v1 in record_params.items():
    for k,v in v1.items():
        if v is None:
            print(f'{k1}-{k} is None!!!')

hessian_num=6 #must < record_test_size,mod 3=0（rgb）
hessian_num_div3 = hessian_num // 3
hessian_images,hessian_labels=None,None
hessian_inputs,tmp_a,tmp_a2=None,None,None
def get_hessian_params():
    a=hessian_images[:,:,10:28,10:28].view((hessian_num,3,3,6,3,6)) #18*18
    a=a[:,:,:,-1:,:,-1:]# 3*3
    tmp_a=a
    tmp_a2 = hessian_images[:, :, 10:28, 10:28].view((hessian_num, 3, 3, 6, 3, 6))  # 18*18
    hessian_inputs=torch.concat((a[:hessian_num_div3,0,...],a[hessian_num_div3:hessian_num_div3*2,1,...],a[hessian_num_div3*2:,2,...]),dim=0)
    hessian_inputs=hessian_inputs.view((-1,))
    return hessian_inputs,tmp_a,tmp_a2
def hessian_regroup(hessian_inputs):
    r,g,b=torch.chunk(hessian_inputs.view((-1,3,1,3,1)),3,dim=0)
    r=torch.concat((r,tmp_a[hessian_num_div3:,0,...]),dim=0)
    g = torch.concat((tmp_a[:hessian_num_div3, 1, ...],g, tmp_a[hessian_num_div3*2:, 1, ...]), dim=0)
    b = torch.concat((tmp_a[:hessian_num_div3*2, 2, ...],b), dim=0)
    group=torch.stack((r,g,b),dim=1)
    a=tmp_a2
    group = torch.concat((a[:, :, :, :5, :, -1:], group), dim=3)
    group = torch.concat((a[:, :, :, :, :, :5], group), dim=-1)
    group=group.view((-1,3,18,18))
    group=torch.concat((hessian_images[:,:,:10,10:28],group,hessian_images[:,:,28:,10:28]),dim=2)
    group = torch.concat((hessian_images[:, :, :, :10], group, hessian_images[:, :, :, 28:]), dim=3)
    return group
#print('regroup right:',torch.allclose(hessian_regroup(hessian_inputs),hessian_images))
def hessian_f(x):
    x = hessian_regroup(x)
    loss = criterion(the_net(x),hessian_labels)
    return torch.mean(loss)
def get_grad_and_hessian():
    if get_grad_hessian:
        the_net.eval()
        hessian_inputs.grad = None
        hessian_inputs.requires_grad = True
        g = torch.autograd.grad(hessian_f(hessian_inputs), hessian_inputs)
        g = torch.abs(g[0])
        g = torch.mean(g)
        hessian_inputs.grad=None
        with torch.no_grad():
            r = torch.autograd.functional.hessian(hessian_f, hessian_inputs, create_graph=False)
            r = torch.abs(r)
            r = torch.diagonal(r)
            r=torch.mean(r)
        return g,r
    else:
        return 0,0
val_s='None'
grad_s='None'
def get_val_grad(grad_r=1):
    global val_s,grad_s
    with torch.no_grad():
        total=0
        val_sum=0
        grad_sum = 0
        for param in the_net.parameters():
            if param.requires_grad and param.grad is not None:
                val_sum+=torch.sum(torch.abs(param.data))
                grad_sum+=torch.sum(torch.abs(param.grad))
                total+=param.data.numel()
        val=val_sum/total
        grad=grad_sum/total
        grad*=grad_r
    val_s='%.2e'%val
    grad_s='%.2e'%grad
def get_test(testloader,test_num=None):
    with torch.no_grad():
        total=0
        correct=0
        the_net.eval()
        total_loss=0
        for data in testloader:
            if test_num and total>test_num:break
            images, labels = data
            if torch.cuda.is_available():
                images=images.cuda()
                labels=labels.cuda()
            # calculate outputs by running images through the network
            outputs = the_net(images)
            loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss * labels.size(0)
    return correct/total,total_loss/total
def get_weights_orth_loss(paras):
    loss = []
    for param in paras:
        if len(param.shape)==4:
            #卷积：[out,in/group,size,size]
            a=param.reshape((param.shape[0],-1))
            a=a/(torch.norm(a,dim=1,keepdim=True).detach()+1e-9)
            m=a@a.T
            loss.append(torch.mean(torch.square(m-torch.eye(param.shape[0]).cuda())))
            pass
        elif len(param.shape)==2:
            #[in,out]
            a = param/(torch.norm(param, dim=0, keepdim=True).detach() + 1e-9)
            m = a.T @ a
            loss.append(torch.mean(torch.square(m - torch.eye(param.shape[1]).cuda())))
    return torch.mean(torch.stack(loss))
def save_grad():
    saved={}
    for name,param in the_net.named_parameters():
        saved[name]=param.grad.clone().detach() if param.grad is not None else None
    return saved
def load_grad(saved):
    for name,param in the_net.named_parameters():
        if param.grad is None or saved[name] is None:param.grad=saved[name]
        else:param.grad[:]=saved[name]
def scale_grad(r):
    for param in the_net.parameters():
        if param.grad is not None:
            param.grad*=r
def work():
    records=[]
    global step,last_show_time,last_save_time,lr,show_lag
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=record_test_size,
                                             shuffle=False, num_workers=1)
    for i, data in enumerate(testloader):
        eval_inputs, eval_labels = data
        eval_inputs = eval_inputs.cuda()
        eval_labels = eval_labels.cuda()
        #print(eval_labels.dtype)
        break

    global hessian_images,hessian_labels,hessian_inputs,tmp_a,tmp_a2
    hessian_images=[]
    hessian_labels=[]
    for i in range(hessian_num):
        hessian_image, hessian_label = trainset[i]
        hessian_images.append(hessian_image)
        hessian_labels.append(hessian_label)
    hessian_images =torch.stack(hessian_images,dim=0)
    hessian_labels = torch.tensor(hessian_labels,dtype=eval_labels.dtype,requires_grad=False)
    hessian_images = hessian_images.cuda()
    hessian_labels = hessian_labels.cuda()
    hessian_inputs, tmp_a, tmp_a2 = get_hessian_params()

    linear_weight_paras=[]
    for name, param in the_net.named_parameters():
        if param.requires_grad:
            if name.endswith('.weight') and len(param.shape) in(2,):
                linear_weight_paras.append(param)
    print('linear weight paras:',len(linear_weight_paras))

    acc=None
    eval_inputs_show=(eval_inputs-torch.amin(eval_inputs))/(torch.amax(eval_inputs)-torch.amin(eval_inputs))
    writer.add_images('record_images',eval_inputs_show,global_step=0)
    loss_mean=None
    acc_train_mean=None
    orth_loss = -1
    batch_size_grouped=0
    canshow_flag_batch_size_group=False
    for epoch_now in range(epochs):
        if epoch_now == 1:
            print('exit warm up, change momentum to:',warm_up_end_momentum)
            change_momentum(warm_up_end_momentum)
        if not use_cos:
            if change_lr and epoch_now==int(epochs*change_lr_epoch_rate1):
                change_momentum(change_momentum1)
                lr = change_lr1
                print('change lr to ',lr,' ,momentum to',change_momentum1)
                net.lr = lr
                change_lr_(optimizer,lr)

            elif change_lr and epoch_now==int(epochs*change_lr_epoch_rate2):
                change_momentum(change_momentum2)
                lr = change_lr2
                print('change lr to ', lr,' ,momentum to',change_momentum2)
                net.lr = lr
                change_lr_(optimizer,lr)
        print(f'new epoch #{epoch_now+1},total:{epochs}')

        tii = time.time()
        for i, data in enumerate(trainloader, 0):
            if i<(20 if epoch_now>0 else 50) or i %(50 if epoch_now>0 else 15)==0 or i==len(trainloader)-1:
                need_record=True
            else:
                need_record=False
            if use_cos and epoch_now >=1:
                scheduler1.step()
                lr=scheduler1.get_last_lr()[0]
                net.lr=lr
            step+=1
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if torch.cuda.is_available():
                inputs=inputs.cuda()
                labels=labels.cuda()
            the_net.train()
            # forward + backward + optimize
            outputs = the_net(inputs)
            loss = criterion(outputs, labels)
            the_loss = loss.detach().data
            if loss_mean is None:
                loss_mean=the_loss
            else:
                r=0.91
                loss_mean=loss_mean*r+(1-r)*the_loss
            ti=time.time()
            if ti-last_show_time>show_lag and canshow_flag_batch_size_group:
                canshow_flag_batch_size_group=False
                test_ti=time.time()
                acc,test_loss=get_test(testloader)
                grad,hessian = get_grad_and_hessian()
                test_time=time.time()-test_ti
                show_lag=test_time*show_lag_div_test_time
                print('change show_lag to %.5f'%show_lag)
                with torch.no_grad():
                    orth_loss = get_weights_orth_loss(linear_weight_paras).detach().data
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels).sum().item()
                    acc_train=correct/len(inputs)
                    if acc_train_mean is None:
                        acc_train_mean = acc_train
                    else:
                        r = 0
                        acc_train_mean = acc_train_mean * r + (1 - r) * acc_train
                writer.add_scalar('loss', the_loss,global_step=step)
                writer.add_scalar('acc', acc, global_step=step)
                writer.flush()
                the_net.train()
                loss.backward()
                batch_size_grouped += len(inputs)
                get_val_grad(grad_r=batch_size/batch_size_grouped)
                print(f"{time.strftime('%Y%m%d-%H:%M:%S',time.localtime(ti))} Epoch:{epoch_now} Step:{step} Loss:{the_loss:.6f} AccTrain:{acc_train_mean:.6f} Acc:{acc:.6f} TestLoss{test_loss:.6f} OrthLoss:{orth_loss:.6f} Grad:{grad:.3e} Hessian:{hessian:.3e} pVal:{val_s} pGrad:{grad_s} Test time:{test_time:.4f}s")

                if batch_size_grouped>=batch_size_group:
                    if batch_size_group>batch_size:print(f'Train a Step,batch_size_grouped:{batch_size_grouped},each batchsize:{batch_size}')
                    scale_grad(batch_size / batch_size_grouped)
                    batch_size_grouped=0
                    canshow_flag_batch_size_group = True
                    if epoch_now > 0:
                        optimizer.step()
                        # scheduler.step()
                    else:
                        # warm up
                        warm_up_optimizer.step()
                    # zero the parameter gradients
                    if epoch_now > 0:
                        optimizer.zero_grad()
                    else:
                        warm_up_optimizer.zero_grad()
                    with torch.no_grad():
                        outputs = the_net(inputs)
                        loss = criterion(outputs, labels)
                    print(f'Loss after step:{loss.data:.6f}')
                with open(step_file,'w',encoding='utf8') as f:
                    f.write(str(step))
                if acc > 0.5:
                    acc_s='%.4f'%acc
                    best_acc_s=best_acc_saved(floats=4)
                    save=lambda :torch.save(the_net.state_dict(), os.path.join(save_dir,f'model-acc{acc_s}-loss{loss_mean:.5f}-testloss{test_loss:.4f}-orthloss{orth_loss:.4f}-acctrain{acc_train_mean:.4f}-grad{grad:.2e}-hessian{hessian:.2e}-pVal{val_s}-pGrad{grad_s}-lr{(lr if epoch_now>0 else warm_up_lr):.1e}-epoch{epoch_now}-step{step}.pth'))
                    if not best_acc_s or best_acc_s<=acc_s:
                        save()
                        print('saved best acc...')
                    elif save_all_show:
                        save()
                        print('saved all show...')
                last_show_time=ti
            else:
                loss.backward()
                batch_size_grouped += len(inputs)
                if batch_size_grouped >= batch_size_group:
                    if batch_size_group>batch_size:print(f'Train a Step,batch_size_grouped:{batch_size_grouped},each batchsize:{batch_size}')
                    scale_grad(batch_size / batch_size_grouped)
                    batch_size_grouped = 0
                    canshow_flag_batch_size_group=True
                    if epoch_now > 0:
                        optimizer.step()
                        # scheduler.step()
                    else:
                        # warm up
                        warm_up_optimizer.step()
                    # zero the parameter gradients
                    if epoch_now > 0:
                        optimizer.zero_grad()
                    else:
                        warm_up_optimizer.zero_grad()
            if ti-last_save_time>save_lag:
                torch.save(the_net.state_dict(),save_path)
                print('saved...')
                last_save_time=ti
            tii2=time.time() #不这样在windows会有bug
            ut=tii2-tii
            tii=tii2
            print(f'global step #{step} ok! use time:{ut:.3f} epoch:{epoch_now} - {((i+1)/len(trainloader)*100):.2f}% loss:{the_loss:.5f} loss mean:{loss_mean:.5f} lr:{lr if epoch_now>0 else warm_up_lr:.6f}')
            if do_record and need_record:
                tiiii=time.time()
                the_net.eval()  # 去掉dropout的随机性
                saved_grad=save_grad()
                optimizer.zero_grad()
                hook = {}
                outputs = the_net(eval_inputs, hook=hook)
                loss = criterion(outputs, eval_labels)
                the_loss = loss.detach().data
                loss.backward()
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    acc_eval=(predicted == eval_labels).sum().item()/eval_labels.size(0)
                    if save_record:record={'step':step,'epoch':epoch_now,'loss':the_loss,'acc_eval':acc_eval,'last_acc':acc,
                            'lr':lr if epoch_now>0 else warm_up_lr,
                            'time':time.strftime('%Y%m%d-%H:%M:%S',time.localtime(tiiii)),
                            'params':{},'features':{}
                            }
                    writer.add_scalar('record_loss', the_loss, global_step=step)
                    writer.add_scalar('record_acc_eval', acc_eval, global_step=step)
                    writer.add_scalar('record_last_acc', acc if acc is not None else 0, global_step=step)
                    writer.add_scalar('record_lr', lr if epoch_now>0 else warm_up_lr, global_step=step)
                    writer.add_scalar('record_epoch', epoch_now, global_step=step)
                    for param_group,params in record_params.items():
                        if save_record:record['params'][param_group]={}
                        for param_name,param in params.items():
                            #print('doing ',param_name)
                            stat_data = util.get_stats(param)
                            stat_grad = util.get_stats(param.grad)
                            stat_grad_div_data = util.get_stats(torch.abs(param.grad/param))
                            if save_record:record['params'][param_group][param_name]=dict(
                                shape=str(param.shape),
                                stat_data=stat_data,
                                stat_grad=stat_grad,
                                data=np.float16(param.cpu().numpy()),
                                grad=np.float16(param.grad.cpu().numpy()),
                            )
                            for II in range(show_scalars):
                                pp=torch.flatten(param)
                                a=(II*31) % pp.shape[0]
                                writer.add_scalar('record_p_'+param_group+'_'+param_name+'_#%d'%a,pp[a], global_step=step)
                                writer.add_scalar('record_p_' + param_group + '_' + param_name+'_#%d'%a+'_grad',torch.flatten(param.grad)[a], global_step=step)
                            for aaa in (stat_data,stat_grad):
                                pre='record_p_'+param_group+'_'+param_name+('_' if aaa is stat_data else '_grad_')+\
                                                   aaa['shape'][aaa['shape'].find('[')+1:-2].replace(' ','')+'_'
                                writer.add_scalar(pre+'pos_mean',aaa['>=0']['mean'],global_step=step)
                                writer.add_scalar(pre + 'pos_std', aaa['>=0']['std'], global_step=step)
                                writer.add_scalar(pre + 'pos_num', aaa['>=0']['num'], global_step=step)
                                writer.add_scalar(pre + 'neg_mean', aaa['<0']['mean'], global_step=step)
                                writer.add_scalar(pre + 'neg_std', aaa['<0']['std'], global_step=step)
                                writer.add_scalar(pre + 'all_std', aaa['all']['std'], global_step=step)
                                writer.add_scalar(pre + 'LT0.1_mean_num', aaa['<mean*0.1']['num'], global_step=step)
                            pre = 'record_p_' + param_group + '_' + param_name +'_GradDivData_'
                            writer.add_scalar(pre + 'all_mean', stat_grad_div_data['all']['mean'], global_step=step)
                            writer.add_scalar(pre + 'all_std', stat_grad_div_data['all']['std'], global_step=step)
                            writer.add_scalar(pre + 'all_max', stat_grad_div_data['all']['max'], global_step=step)
                            if param_group=='conv_weight':
                                pre = f'record_pConv_margin{weight_margin:.1f}_' + param_group + '_' + param_name
                                img=(param.data[:2048,0:1,:,:]+weight_margin)/(2*weight_margin)
                                writer.add_images(pre,img,global_step=step,dataformats='NCHW')
                                pre = f'record_pConv_relative_' + param_group + '_' + param_name
                                minn=torch.amin(param.data[:2048, 0:1, :, :])
                                img = (param.data[:2048, 0:1, :, :] -minn) / (torch.amax(param.data[:2048, 0:1, :, :])-minn)
                                writer.add_images(pre, img, global_step=step,dataformats='NCHW')
                            elif len(param.shape)==2:
                                pre = f'record_pLinar_margin{weight_margin:.1f}_' + param_group + '_' + param_name
                                img = (param.data + weight_margin) / (2 * weight_margin)
                                writer.add_image(pre,img, dataformats='HW', global_step=step)
                                pre = f'record_pLinar_relative_' + param_group + '_' + param_name
                                minn = torch.amin(param.data)
                                img = (param.data - minn) / (torch.amax(param.data) - minn)
                                writer.add_image(pre,img, dataformats='HW', global_step=step)
                    for key in hook:
                        # print(hook[key].shape)
                        map=hook[key]
                        if len(map.shape)==2:
                            map=map[:,:,None,None]
                        stat = util.get_stats_feature_map(map,thres=channel_thres)
                        if save_record:record['features'][key] = dict(
                            shape=str(map.shape),
                            stat=stat,
                            data=np.float16(map.cpu().numpy()),
                        )
                        pre='record_f_' +  key  +stat['shape'][stat['shape'].find('[') + 1:-2].replace(' ', '')+'_'
                        writer.add_scalar(pre + 'global_mean',stat['global_mean'], global_step=step)
                        writer.add_scalar(pre + 'global_std', stat['global_std'], global_step=step)
                        writer.add_scalar(pre + 'channel_mean_max', stat['channel_mean_max'], global_step=step)
                        writer.add_scalar(pre + 'channel_mean_min', stat['channel_mean_min'], global_step=step)
                        writer.add_scalar(pre + 'channel_max_min', stat['channel_max_min'], global_step=step)
                        writer.add_scalar(pre + 'channel_std_mean', stat['channel_std_mean'], global_step=step)
                        writer.add_scalar(pre + 'channel_std_min', stat['channel_std_min'], global_step=step)
                        writer.add_scalar(pre + 'channel_margin_mean', stat['channel_margin_mean'], global_step=step)
                        writer.add_scalar(pre + 'channel_margin_min', stat['channel_margin_min'], global_step=step)
                        writer.add_scalar(pre + 'map_std_mean', stat['map_std_mean'], global_step=step)
                        writer.add_scalar(pre + 'map_margin_mean', stat['map_margin_mean'], global_step=step)
                        writer.add_scalar(pre + "channel_died_%.2f"%channel_thres, stat["channel_died_%.2f"%channel_thres], global_step=step)
                        writer.add_scalar(pre + "channel_margin_die_%.2f"%channel_thres, stat["channel_margin_die_%.2f"%channel_thres], global_step=step)
                        writer.add_scalar(pre + "channel_died2_%.2f"%channel_thres, stat["channel_died2_%.2f"%channel_thres], global_step=step)
                        if map.shape[2]>1 and map.shape[3]>1:
                            for I in range(feature_map_show_num):
                                m=map[:,I*3:I*3+3,:,:]
                                m=m/torch.amax(m)
                                writer.add_images(f'record_f{I*3}-{I*3+2}_' +  key  +
                                stat['shape'][stat['shape'].find('[') + 1:-2].replace(' ', ''),m, global_step=step)
                        else:
                            m = map[:,:, 0, 0]
                            m = m / torch.amax(m)
                            writer.add_image(f'record_f_all_' + key +
                                              stat['shape'][stat['shape'].find('[') + 1:-2].replace(' ', ''),
                                              m, dataformats='HW', global_step=step)
                if save_record:records.append(record)
                print('recorded ut:%.5f acc_eval:%.5f'%(time.time()-tiiii,acc_eval))
                load_grad(saved_grad)
        writer.flush()
    with open(step_file, 'w', encoding='utf8') as f:
        f.write(str(step))
    writer.flush()
    if save_record:np.save(record_save_path,records)
    torch.save(the_net.state_dict(), save_path)
    print('saved...')
    acc,test_loss = get_test(testloader)
    grad,hessian=get_grad_and_hessian()
    print(f'final acc:{acc:.10f} ,test loss:{test_loss:.10f},grad:{grad:.3e},hessian:{hessian:.3e}')
    if acc > 0.5:
        acc_s = '%.4f' % acc
        best_acc_s = best_acc_saved(floats=4)
        save = lambda: torch.save(the_net.state_dict(), os.path.join(save_dir,
                                                                     f'model-acc{acc_s}-loss{loss_mean:.5f}-testloss{test_loss:.4f}-orthloss{orth_loss:.4f}-acctrain{acc_train_mean:.4f}-grad{grad:.2e}-hessian{hessian:.2e}-pVal{val_s}-pGrad{grad_s}-lr{(lr if epoch_now > 0 else warm_up_lr):.1e}-epoch{epoch_now}-step{step}.pth'))

        if not best_acc_s or best_acc_s <= acc_s:
            save()
            print('saved best acc...')
        elif save_all_show:
            save()
            print('saved...')

if __name__ == '__main__':

    work()
