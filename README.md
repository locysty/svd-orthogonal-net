###  Improve performance with weight SVD orthogonalization and BatchNorm bias init to 1
####  weight SVD orthogonalization:
Linear and depthwise conv (1*1) layer can be seen as matrix matmul. When the weights are orthogonal, the correlation of the output units is reduced and the diversity is increased. And when backpropagating, the norm of the gradient will not increase, avoiding explosion or disappearance

There are parameters orthogonal init ways in pytorch : nn.init.orthogonal_().
But I find initialing weight orthogonally get worse performance in testing cifar10. Because the orthogonality decreases rapidly after training, even worse than the orthogonality from normal initialization

So I try svd decomposition to force the weights to have to be orthogonalized:
```
#weight is depthwise conv layer weight
weight.requires_grad=False
out_,in_,s1,s2 =weight.shape
a=weight.reshape((self.groups,-1,(in_*s1*s2))) #self.groups is nn.Conv2D 'groups' param in pytorch
a=a.transpose(1,2)
u, s, vh = torch.linalg.svd(a, full_matrices=False)
mat = u @ vh
if a.shape[2] > a.shape[1]:
    mat = mat / (torch.linalg.norm(mat, axis=0, keepdims=True).detach() + 1e-10)
mat=mat.transpose(1,2)
weight[:] = mat.reshape((out_,in_,s1,s2))
weight.requires_grad=True
```
Note: I reassign weight values at the end of the code. I have tested without assignment, and the effect is not as good as assigning. And there is no need to calculate gradients during backpropagation with value assignment

Using this, the effect is basically improved in various situations ,without increasing model inference time (can remove the svd code after training)

cifar10 test result:

| Model size  | With SVDOrth  |  Optimizor | Seed A accuracy  | Seed B accuracy  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| small(175k)  | **yes**  |  SGD |  79.6 | 79.6|
| small | no | SGD | 80.5 | 79.9 |
|small	|**yes**	|Adamw	|78.3|	78.6|
|small|	no|	Adamw|	77.5|	76.7|
|large(3.2m)|	**yes**|	SGD|	87.0|	86.8|
|large|	no|	SGD|	84.9|	85.3|
|large|**yes** |	Adamw | 87.1|	87.1|
|large|	no	|Adamw|	84.0|	84.3|

also I test in a 46cls album multi classification task with efficientnet-b0 in my actual work.The recall improve 3-5% with weight SVD orthogonalization

####  BatchNorm bias init to 1:
when I look into many BatchNorm bias values after training.It seems that most of the bias are positive numbers in better accuracy model

I guess because: Usually the layer order is Conv -> BatchNorm -> Relu . 
And BatchNorm -> Relu = Statistically normalize -> affine -> Relu

if bias in affination is more positive, more information will be retained after passing through the relu layer, the module is more linear for better learning

But if the bias is too large, this module will degenerate into a completely linear, unable to extract features

I've test these with cifar10 ,found bias init to 1 is slightly better compared to 0 or 2,with accuracy improved by 1-2%

Here is sample train code with cifar10. Please run train_tiny_record.py 
