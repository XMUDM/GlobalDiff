import enum
import math
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn





# 定义了一个名为 ModelMeanType 的枚举类，其中包含了两个选项：START_X 和 EPSILON。
# 对于每个选项，使用 enum.auto() 来自动生成一个独特的值，以代表不同的选项。
# 这个枚举类可以用于在代码中表示模型预测的均值类型。
class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

# 高斯扩散（Gaussian Diffusion）模型
class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device,item_num, history_num_per_term=10, beta_fixed=True):
        
        # 均值类型
        self.mean_type = mean_type
        # 用于控制每个扩散步骤中噪声的变化方式。它定义了在每个步骤中添加到输入数据中的噪声的规模，从而在每个步骤中逐渐扩散数据。
        self.noise_schedule = noise_schedule
        # noise_scale 是高斯扩散模型中的一个参数，用于控制添加到数据中的噪声的初始强度。它影响每个扩散步骤中添加到数据中的噪声的大小。
        self.noise_scale = noise_scale
        #最小噪声
        self.noise_min = noise_min
        #最大噪声
        self.noise_max = noise_max
        # 表示扩散过程的步数。它指定了在生成样本时模型将数据逐步扩散的步骤数量。
        self.steps = steps
        #模型设备
        self.device = device
        # token id数目
        self.item_num =item_num
        # history_num_per_term 参数指定了在每个步骤中保留的历史数据的数量。这些历史数据可能用于计算噪声的标准差等参数，以及在模型的后续步骤中进行更新。
        # 选择适当的 history_num_per_term 可能会影响模型生成的样本质量和稳定性。较大的值可能会使模型更稳定，但也可能导致计算开销增加
        self.history_num_per_term = history_num_per_term
        
        # 用于在高斯扩散模型中存储每个步骤的历史数据。这个张量的形状是 (steps, history_num_per_term)，其中 steps 是扩散过程的总步数，而 history_num_per_term 是每个步骤中保留的历史数据数量。
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        # Lt_count 张量将用于记录每个步骤中历史数据的计数，以便在模型的运行过程中进行更新。具体来说，当在一个步骤中添加新的历史数据时，相应的计数将增加。
        self.Lt_count = th.zeros(steps, dtype=int).to(device)
   
        #如果噪声的初始强度不为0
        if noise_scale != 0.:
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            # beta要是1D张量
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            #beta数组的长度要与扩散步骤相同
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            # beta要大于0小于1
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"
            # 计算高斯噪声
            self.calculate_for_diffusion()

        super(GaussianDiffusion, self).__init__()
        #交叉熵损失
        self.ce4diff = nn.CrossEntropyLoss(reduction='none')
        #交叉熵损失,不计算label 为0的item的loss
        self.ce4bert = nn.CrossEntropyLoss(ignore_index=0)
    # 创建beta，beta是一个1D张量，形状为[steps]
    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        # 如果beta是线性的
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            # 根据线性变化的标准差计算扩散模型中每个步骤的 betas
            if self.noise_schedule == "linear":
                 # 使用 np.linspace() 创建一个线性间隔的数组，该数组的起始值是 self.noise_scale * self.noise_min，结束值是 self.noise_scale * self.noise_max，
                # 元素数量为 self.steps，表示高斯噪声在每个步骤中逐步增加。
                return np.linspace(start, end, self.steps, dtype=np.float64)
            
            else:
                # 根据线性变化的方差计算扩散模型中每个步骤的 betas
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            #基于cosine的扩散序列
            return betas_for_alpha_bar(self.steps,lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
        
        # "binomial" 噪声调度方式使用一种递减的规律来计算每个步骤的 beta 值。
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    # 计算高斯噪声
    def calculate_for_diffusion(self):
        #获得alpha权重，alpha为1D张量
        alphas = 1.0 - self.betas
        # 计算给定的 alphas 数组的累积乘积，并将结果存储在名为 alphas_cumprod 的张量中。
        # th.cumprod(input, dim=None) 函数接受以下参数：
        # input: 输入的 Pyth 张量，可以是任意形状和数据类型。
        # dim: 指定进行累积操作的维度。如果不提供此参数，则会对整个张量进行累积。如果指定了 dim，则沿着指定的维度进行累积操作。
        # 函数的返回值是一个张量，其形状与输入张量相同，但包含了在每个位置上的累积乘积结果。
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        # 在 alphas_cumprod 前面添加一个值为 1.0 的张量，即当前时间步长之前的 (1-beta) 的累积乘积。
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        # 后面添加一个值为 0.0 的张量，即当前时间步长之后的 (1-beta) 的累积乘积。
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
       
        assert self.alphas_cumprod_prev.shape == (self.steps,)
        # 在高斯扩散模型中计算每个时间步长上 (1-beta) 的累积乘积的平方根，以供后续的模型计算使用。
        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        # 在高斯扩散模型中计算每个时间步长上 (1 - alpha) 值的平方根，以供后续的模型计算使用。
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        # 在高斯扩散模型中计算每个时间步长上 (1 - alpha) 值的log，以供后续的模型计算使用。
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        # 在高斯扩散模型中计算每个时间步长上 (1 / alpha) 值的平方根，以供后续的模型计算使用。
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        # 在高斯扩散模型中计算每个时间步长上 (1 / alpha-1) 值的平方根，以供后续的模型计算使用。
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)
        # 在高斯扩散模型中计算每个时间步长上 (1 -每个时间步骤之前的alpha/每个时间步骤上的alpha)，以供后续的模型计算使用。
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 首先，从后验方差中取出第二个元素（索引为 1）作为新的张量。
        # 然后，将这个新的张量与后验方差的其他元素拼接在一起，形成一个新的张量。
        # 最后，对这个新的张量进行取对数操作，得到一个新的张量
        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        # 计算高斯噪声模型中的后验均值的一个系数。
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 将当前时间点之前的累积乘积与 alphas 的平方根相乘，用于调整均值。
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * th.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    #接受逼近器，一个batch的数据[batch_size,item_num],前向推断的步数，以及是否抽样噪声
    # 在扩散模型中进行概率采样并作推断
    def p_sample(self, model, x_start, seqs , mask_indice,steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        # 前向传播
        # x_start 表示采样的起始状态，即初始的扩散状态。如果需要进行 0 步采样，那么结果就是直接使用 x_start 作为样本路径的结束状态
        if steps == 0:
            x_t = x_start
        else:
            # 首先，通过创建一个包含 steps - 1 的列表，用于构建一个时间步长张量 t。该时间步长张量的形状为 (batch_size,)，其中 batch_size 表示样本数量。
            # 接着，调用 self.q_sample(x_start, t) 方法，使用初始状态 x_start 和时间步长张量 t 进行采样，得到一个初始的样本路径 x_t。
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        # 创建了一个从 0 到 self.steps - 1 的索引列表，然后 [::-1] 操作将这个列表进行逆序排列。得到的 indices 列表将按逆序遍历的顺序包含所有时间步长的索引。
        indices = list(range(self.steps))[::-1]
        #不考虑噪声强度，利用模型逼近并获得最终的从加噪还原step步的原始输入
        if self.noise_scale == 0.:
            for i in indices:
                #t为逆序的时间步长，例如从5-4-3-2-1.形状依然为[batch_size]
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t , seqs_logits,diff_logits,inputs_encoding= model(x_t, seqs, t,mask_indice)
            return x_t.squeeze(1)
        
        # 代码根据 indices 列表中的逆序时间步长索引，进行采样过程。
        for i in indices:
            # 根据当前时间步长 i 创建一个时间步长张量 t。
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            # 计算在当前状态 x_t 和时间步长 t 下的均值和方差等的字典
            out = self.p_mean_variance(model, x_t, seqs, t,mask_indice)
            # 在扩散模型中，推断过程涉及对噪声的使用，以及根据后验分布的均值和方差进行样本采样。
            if sampling_noise:
                # 生成与 x_t 形状相同的随机噪声 noise [batch_size,item_num]
                noise = th.randn_like(x_t)
                # 当前的时间步长 t，生成一个与 x_t 形状相同的非零噪声掩码 nonzero_mask，以确保在 t 为 0 时不添加噪声。
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                # 将均值 out["mean"] 与噪声相结合，通过添加 nonzero_mask 乘以标准差的一半和随机噪声的乘积来模拟噪声采样，从而得到一个新的状态 x_t。
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                # 直接使用均值 out["mean"] 作为新的状态 x_t。
                x_t = out["mean"]
        # [batch_size,item_num]
        return x_t.squeeze(1)

    def Target2OneHot(self,target):
        B,L = target.size()
        #token id 是从1开始算，所以最大的token id 是 item_num
        # 创建一个全零的张量，形状为(B, N)，其中N为类别数量
        one_hot = th.zeros(B, self.item_num+1).to(target.device)
        one_hot[th.arange(B).unsqueeze(1),target]=1
        # (B,1,N)
        return one_hot.unsqueeze(1)
    
    # 正向反向传播。model:逼近器  x_start : batch数据[batch_size,item_num]（x0）
    # def p_losses(self, denoise_model, x_start,h,Nex_start,seqs_labels,seq_logits,loss_type="cl-mse" ,reweight=False):
    def p_losses(self, denoise_model, seqs,seqs_labels,target,target_logits,negative_target,random_indice,curr_epoch, noise=None, loss_type="ce",reweight=False):
        
        """
        扩散模型的前向传播
            args:
                denoise_model: 逼近器
                seqs (torch.tensor (B,S+1)) : 序列数据，形状为 (B, S+1)，其中 B 是批处理大小,S 是序列长度。这里的 S+1 表示序列长度加一,因为序列数据还包括一个global标记。
                seqs_labels (torch.tensor (B,S)): 序列标签数据，形状为 (B, S)。
                target (torch.tensor (B,1)): 目标数据，形状为 (B, 1)。
                negative_target (torch.tensor (B,K)): 负样本目标数据，形状为 (B, K)。
                random_indice (torch.tensor (B,1)): target item在seqs中的索引,形状为 (B, 1)。
                curr_epoch (int) : 当前训练轮数。
                loss_type (str): 损失函数类型，可以是 "l1" 或 "l2"。

        """
        #将目标转成独热向量 (B,1,V)
        # x_start =self.Target2OneHot(target.unsqueeze(1))
        # 将target的模拟logits扩展一个维度
        x_start=target_logits.unsqueeze(1)

        
        batch_size, device = x_start.size(0), x_start.device
        #ts 是生成的时间步长索引，而 pt 是生成的概率分布，用于确定每个时间步骤的采样概率。ts，pt形状为[batch_size]
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        # 建了一个与给定张量 x_start 具有相同形状的随机噪声张量 noise，并且每个元素都是从标准正态分布中生成的随机数。
        noise = th.randn_like(x_start)
        #判断噪声的初始强度，进行扩散模型的前向传播
        if self.noise_scale != 0.:
            # x_t为进行ts规定的K个时间步长后的加噪结果，形状为[batch_size,1,item_num]
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start
        #(B,1,V),K个负样本
        Nex_start=self.Target2OneHot(negative_target)
        totalloss=0
        CeLoss=0
        CeLoss4bert=0
        CeLoss4diff=0


        model_output,seqs_logits,diff_logits,inputs_encoding= denoise_model(x_t,seqs,ts,random_indice)
        # 使用一个字典来根据给定的 self.mean_type 值选择不同的键对应的值，并将选定的值赋给变量 target。
        predict_target = {
            #对逼近器x_0
            ModelMeanType.START_X: x_start,
            #对所加的噪声
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        #再次判断形状相同
        assert model_output.shape == predict_target.shape == x_start.shape

        if loss_type == 'l1':
            loss = F.l1_loss(x_start, model_output)
        elif loss_type == 'l2':
            #mse 
            loss = F.mse_loss(x_start, model_output, reduction='none').sum(dim=-1).mean(dim=-1)
        elif loss_type == 'mse':
            #mse 
            loss = F.mse_loss(x_start, model_output, reduction='none').sum(dim=-1).mean(dim=-1)
            #编码器的交叉熵损失
            labels=seqs_labels.view(-1)
            CeLoss4bert=self.ce4bert(seqs_logits.to(labels.device),labels)
            totalloss=loss +  CeLoss4bert

        elif loss_type=='BPR':
            #diff bpr loss
            positive_logitis=diff_logits[th.arange(target.size(0)),target]
            negative_logitis=diff_logits[th.arange(target.size(0)).unsqueeze(1),negative_target]
            positive_logitis=positive_logitis.unsqueeze(1).repeat(1,negative_logitis.size(1))
            BPRloss=-th.log(th.sigmoid((positive_logitis-negative_logitis))).mean(dim=1)

            #diff的交叉熵损失
            CeLoss4diff=self.ce(diff_logits.to(self.device),target)
            totalloss=CeLoss4diff + BPRloss 

        elif loss_type=='ce':
            #编码器的交叉熵损失
            labels=seqs_labels.view(-1)
            CeLoss4bert=self.ce4bert(seqs_logits.to(labels.device),labels)
            #diff的交叉熵损失
            CeLoss4diff=self.ce4diff(diff_logits.to(labels.device),target)
            # totalloss=CeLoss4diff + CeLoss4bert
            totalloss =CeLoss4diff


        #为不同的时间步长分配不同权重
        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                # 计算相邻时间步长的信噪比之差，然后将该差值赋给变量 weight。这种做法可能是为了根据信噪比之差来调整模型在不同时间步长上的权重。具体来说，如果信噪比之差较大，可能意味着信号相对噪声更强，因此在这个时间步长上赋予更大的权重。
                weight = self.SNR(ts - 1) - self.SNR(ts)
                # th.where(condition, x, y): 这是 Pyth 中的函数，根据给定的条件 condition，在 x 和 y 之间选择相应的值。如果 condition 为 True，则选择 x；如果 condition 为 False，则选择 y。
                weight = th.where((ts == 0), 1.0, weight)
                totalloss = totalloss
            #?没读懂
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                totalloss = th.where((ts == 0), likelihood, totalloss)
        #不为不同的时间步长分配不同权重
        else:
            weight = th.tensor([1.0] * len(target)).to(device)
        #记录loss,形状为[batch_size]
        totalloss = weight * totalloss
        # update Lt_history & Lt_count
        # 更新模型中的历史损失记录
        for t, loss in zip(ts, totalloss):
            # 如果每个步骤中保留的历史数据的数量与每个步骤中历史数据的计数相同
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                # 更新历史损失记录 Lt_history，将旧记录往前移动一个位置，然后在最后一个位置上设置为当前时间步长的损失值 loss。
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            # 如果每个步骤中保留的历史数据的数量与每个步骤中历史数据的计数不同
            else:
                try:
                    # 在每个时间步长的历史记录中按顺序添加损失值。
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError
        # 目的是将每个时间步长的损失值进行除法操作，将其除以变量 pt 的值。这是为了根据时间步长的权重来调整损失值。具体地说，如果某个时间步长的权重较大，那么对应的损失值将会减小。
        totalloss /= pt
        return totalloss.mean(),model_output


    
    # 生成时间步长和对应的概率
    # 生成的时间步长索引 t 是用于确定在模型中执行哪些时间步骤。具体来说，生成的时间步长索引 t 决定了在扩散模型中应用多少个时间步骤来进行推断和预测。
    # 生成的概率分布 pt 用于指定每个时间步长被选择的概率。这个概率分布对应于生成的时间步长索引 t，从而确定了每个时间步骤在模型中的应用程度。
    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            # 如果每个步骤中保留的历史数据的数量与每个步骤中历史数据的计数不同
            if not (self.Lt_count == self.history_num_per_term).all():
                # 生成时间步长和对应的概率
                return self.sample_timesteps(batch_size, device, method='uniform')
            # 将历史记录的平方根的平均值的平方根计算出来，以用于后续生成时间步长的概率分布。
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            # 计算了概率分布中每个时间步长的概率，将 Lt_sqrt 的平方根的平均值除以所有平方根的总和，以确保它们之和为1。
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            # 将每个时间步长的概率与 1 - uniform_prob 相乘，从而降低了均匀采样的权重，增加了根据历史记录的权重。
            pt_all *= 1- uniform_prob
            # 将均匀采样的概率分布添加到每个时间步长的概率中，从而混合了历史记录的权重和均匀采样的权重。
            pt_all += uniform_prob / len(pt_all)
            # 用于验证概率分布 pt_all 的总和是否接近1。
            assert pt_all.sum(-1) - 1. < 1e-5
            # 使用多项式分布抽样，根据给定的概率分布 pt_all 生成一批时间步长。t 是一个包含多项式分布随机样本的张量，表示生成的时间步长。每个样本的值表示选中的时间步长索引。
            # pt_all: 这是之前计算得到的概率分布，表示每个时间步长的概率。
            # num_samples: 这是要生成的随机样本数量，即时间步长的数量。
            # replacement=True: 这是一个布尔值参数，表示是否可以重复抽样。如果为 True，则允许重复抽样；如果为 False，则不允许重复抽样。
            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            # 生成的 pt 是一个张量，表示每个生成的时间步长对应的概率。
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            #t，pt都为[batch_size]的1D张量
            return t, pt
        
        elif method == 'uniform':  # uniform sampling

            # th.randint(0, self.steps, (batch_size,), device=device): 
            # 使用 randint 函数生成随机整数张量，其中参数分别为下限（0）、上限（self.steps）和生成的随机数数量（batch_size）。
            # 这将生成一个形状为 (batch_size,) 的整数张量，表示生成的时间步长索引。
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            # 表示与时间步长索引 t 相同形状的概率分布。由于所有元素都设置为1，表示每个时间步长的概率都是1，即均匀分布。这种方式是将概率分布设置为均匀的，而不考虑特定的概率权重。
            pt = th.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError
    # 用于在扩散模型中进行采样操作。给定初始数据 x_start、时间步长索引 t 和可选的噪声 noise，该函数会根据扩散模型的公式生成采样数据。
    # 对应于扩散模型正向传播公式
    # x_start[batch_size,item_num],t[batch_size]
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            # 第一个部分使用 self.sqrt_alphas_cumprod 对 x_start 进行逐元素乘法，然后使用 _extract_into_tensor 函数根据时间步长索引 t 提取对应的乘法因子。
            # 第二个部分使用 self.sqrt_one_minus_alphas_cumprod 对噪声 noise 进行逐元素乘法，然后使用 _extract_into_tensor 函数根据时间步长索引 t 提取对应的乘法因子。
            # 最终，两个部分的结果进行逐元素加法，得到最终的采样数据。
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    # 用于计算扩散模型后验分布的均值和方差。具体来说，它计算了后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差。
    # x_start 表示起始状态，x_t 表示当前状态，t 表示时间步长。
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        # 方法首先根据给定的时间步长 t 抽取出对应时间步长的系数，然后使用这些系数对起始状态和当前状态进行加权求和，从而计算出后验分布的均值。
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 方法也从相应的数组中抽取对应时间步长的后验方差和对数方差信息
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        # 得到的后验分布的均值 posterior_mean、方差 posterior_variance 和经过剪裁的对数方差 posterior_log_variance_clipped 作为结果返回。
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    #接受逼近器，加噪x[batch_size,item_num],时间步长索引t[batch_size]
    # 用于对模型进行推断，以获取条件概率分布的均值和方差
    def p_mean_variance(self, model, x, seqs, t,mask_indice):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        # 确保第x，t一个维度相同
        B, S = x.shape[:2]
        assert t.shape == (B, )
        #模型逼近输出
        # model_output = model(x, h, t)
        model_output , seq_logits,diff_logits,inputs_encoding= model(x, seqs, t, mask_indice)
        # 将 self.posterior_variance 的值赋给了变量 model_variance。根据之前的代码和上下文，self.posterior_variance 是一个包含了后验方差信息的张量，用于模拟扩散模型中的方差部分。
        model_variance = self.posterior_variance
        # self.posterior_log_variance_clipped 是一个包含了后验方差的对数值（经过剪裁操作后）的张量，用于模拟扩散模型中的对数方差部分。
        model_log_variance = self.posterior_log_variance_clipped
        # 将 model_variance 中的特定时间步长 t 对应的值抽取出来，并根据输入张量 x 的形状进行广播，得到一个与 x 形状相同的新张量，作为模型的方差信息。
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        # 将 model_log_variance 中的对数方差值根据特定的时间步长 t 进行抽取，并进行广播以适应输入张量 x 的形状。这样可以在推断过程中使用对应时间步长的对数方差信息。
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        #直接将模型的输出作为加噪数据还原prediction的结果[batch_size,item_num]
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        #？忽略了
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        # 用于计算扩散模型后验分布的均值和方差。具体来说，它计算了后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差。
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        # 返回模型的均值，方差，方差对数，还原加噪的字典
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    # 计算模型中单个时间步长的信噪比
    #t形状为[batch_size]
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        # ：信噪比 = α_cumprod[t] / (1 - α_cumprod[t])，其中 α_cumprod[t] 表示从起始到给定时间步长 t 的累积乘积。
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    # 用于从一个 1-D 张量中根据给定的索引生成一个新的张量，其中索引由参数 timesteps 指定。
    # 这个函数的目的是根据索引生成一个与输入形状相同的新张量，其中每个位置的值来自于输入张量相应索引位置的值。
    # arr为[step],timesteps为[batch_size],broadcast_shape为[batch_size,item_num]
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()

        # 将输入张量 arr 移动到与 timesteps 相同的设备上。
        arr = arr.to(timesteps.device)
        # 函数根据 timesteps 的索引从 arr 中提取对应位置的值，并将结果转换为浮点数类型。
        res = arr[timesteps].float()
        # 将提取的结果在末尾逐步添加新的维度，直到达到目标形状的维度数量，从而使得提取的结果的形状与目标形状相同。
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        # 函数使用 expand 函数将提取的结果进行扩展，使其形状与目标形状相同。扩展将在缺少的维度上复制数据，以匹配目标形状
        return res.expand(broadcast_shape)
# 根据线性变化的方差计算扩散模型中每个步骤的 betas
# 接受三个参数：steps（步骤数量）、variance（线性变化的方差数组）和 max_beta
def betas_from_linear_variance(steps, variance, max_beta=0.999):
    # 使用递推方式计算每个步骤的 beta 值。首先，根据方差数组计算每个步骤中的 alpha_bar（1 减去方差）值。
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        # 控制高斯噪声的标准差
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

#基于cosine的扩散序列
# 接受三个参数：num_diffusion_timesteps（扩散过程的总步数）、alpha_bar（一个函数，定义了在每个时间步长上 (1-beta) 的累积乘积）、max_beta（最大允许的 beta 值）
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

# 用于计算张量在除批次维度外的所有维度上的平均值。
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
