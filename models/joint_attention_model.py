from collections import OrderedDict

import torch
from torch import nn
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

class JointAttentionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(norm='batch', netG='mh_resnet_6blocks', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.n_input_modal = opt.n_input_modal
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_cls', 'G_UC']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        self.attention_type = opt.attention_type
        self.netG = networks.define_MHG(opt.attention_type, opt.n_input_modal, opt.input_nc+opt.n_input_modal+1, opt.output_nc, opt.ngf, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type,
                                          opt.init_gain, self.gpu_ids, len(opt.modal_names))

        if self.isTrain:
            # define loss functions
            self.criterionCls = nn.CrossEntropyLoss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionL2 = torch.nn.MSELoss()
        self.all_modal_names = opt.modal_names
        self.l_gradient_penalty = opt.l_gradient_penalty
        self.n_cls = opt.n_input_modal + 1

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_B_no_mask = input['B'][:, :self.opt.input_nc].to(self.device)
        self.modal_names = [i[0] for i in input['modal_names']]

        target_modal_names = input['modal_names'][-1]
        self.real_B_Cls = torch.tensor([self.all_modal_names.index(i) for i in target_modal_names]).to(self.device)
        # self.modal_names = list(map(lambda x: x[0], input['modal_names']))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def Cls(self):
        with torch.no_grad():
            _, cls_real = self.netD(self.real_B_no_mask)
            score, predicted = torch.max(cls_real, 1)
            correct = (predicted == self.real_B_Cls).sum().item()
            return correct
    
    def Unc(self, i):
        with torch.no_grad():
            _, cls_real = self.netD(i)
            predicted, _ = torch.max(cls_real, 1)
            return predicted

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B.detach()  # fake_B from generator
        g_pred_fake, g_cls_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(g_pred_fake, False)

        # Real
        if self.l_gradient_penalty > 0:
            self.real_B_no_mask.requires_grad_(True)
        pred_real, cls_real = self.netD(self.real_B_no_mask)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # self.loss_D_cls = self.criterionCls(cls_real, self.real_B_Cls)
        y = F.one_hot(self.real_B_Cls, self.n_cls).float()
        # print(y)
        self.loss_D_cls = self.bce_loss_fn(cls_real, y)
        self.loss_D_cls += self.l_gradient_penalty * self.calc_gradient_penalty(self.real_B_no_mask, cls_real)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real + 5 * self.loss_D_cls) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        g_pred_fake, g_cls_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(g_pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B_no_mask) * self.opt.lambda_L1

        y = F.one_hot(self.real_B_Cls, self.n_cls).float()
        self.loss_G_UC = self.bce_loss_fn(g_cls_fake, y)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_UC

        self.loss_G.backward()

    def update_embeddings(self):
        y = F.one_hot(self.real_B_Cls, self.n_cls).float()
        # print(self.netD.module.N)
        self.netD.module.N = self.netD.module.gamma * self.netD.module.N + (1 - self.netD.module.gamma) * y.sum(0)
        features = self.netD.module.model(self.real_B_no_mask)
        # features = features.view(features.size(0), -1)
        cls = self.netD.module.cls_branch(features)
        b, c, h, w = cls.shape
        z = cls.view([b, c])

        z = torch.einsum("ij,mnj->imn", z, self.netD.module.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.netD.module.m = self.netD.module.gamma * self.netD.module.m + (1 - self.netD.module.gamma) * embedding_sum

    def bce_loss_fn(self, y_pred, y):
        bce = F.binary_cross_entropy(y_pred, y, reduction="sum").div(
            self.n_cls * y_pred.shape[0]
        )
        return bce

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        # self.netD.train()
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        self.real_B_no_mask.requires_grad_(False)
        with torch.no_grad():
            self.netD.eval()
            self.update_embeddings()
        self.netD.train()
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_visuals(self):
        """Calculate additional output images for tensorboard visualization"""
        pass

    def get_current_visuals(self):
        modal_imgs = []
        for i in range(self.n_input_modal):
            modal_imgs.append(self.real_A[:, i*(self.n_input_modal+1+self.opt.input_nc):i*(self.n_input_modal+1+self.opt.input_nc)+self.opt.input_nc, :, :])
        modal_imgs.append(self.real_B_no_mask)
        visual_ret = OrderedDict()
        for name, img in zip(self.modal_names, modal_imgs):
            visual_ret[name] = img
        visual_ret['fake_' + self.modal_names[-1]] = self.fake_B

        return visual_ret

    def get_encoder_features(self):
        encoder_features, attention_features = self.netG.module.get_features(self.real_A)
        if attention_features is not None:
            attention_features = torch.chunk(attention_features, self.opt.n_input_modal, dim=1)
        if hasattr(self, 'sr'):
            sr_encoder_features = self.sr.get_features(self.real_B)
        else:
            sr_encoder_features = None
        return encoder_features, attention_features, sr_encoder_features

    def calc_gradients_input(self, x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]
        
        gradients = gradients.flatten(start_dim=1)
        # print(y_pred.size())

        return gradients

    def calc_gradient_penalty(self, x, y_pred):
        gradients = self.calc_gradients_input(x, y_pred)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty