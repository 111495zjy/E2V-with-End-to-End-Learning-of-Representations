import torch.nn as nn
from .base_layers import *

class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 2, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 4
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

    def forward(self, x):
        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))
        x = self.mlp[-1](x)
        x = x.squeeze()
        return x




class QuantizationLayer(nn.Module):
    def __init__(self, num_bins = 5, height = 100, width = 100,mlp_layers=[4, 40, 40, 1],activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,activation=activation,num_channels=num_bins)
        self.num_bins = num_bins
        self.width = width
        self.height = height
    def forward(self, events):
            
        assert(events.shape[1] == 4)
        assert(self.num_bins > 0)
        assert(self.width > 0)
        assert(self.height > 0)

        voxel_grid = torch.zeros(self.num_bins, self.height, self.width, dtype=torch.float32, device=events.device).flatten()

        if len(events) == 0:
             return voxel_grid.view(1,self.num_bins, self.height, self.width)
          # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp
        if deltaT == 0:
              deltaT = 1.0
        events[:, 0] = (events[:, 0] - first_stamp) / deltaT
        ts = events[:, 0].float()
        xs = events[:, 1].long()
        ys = events[:, 2].long()
        pols = events[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        idx_before_bins = xs  + self.width * ys

        for i_bin in range(self.num_bins):
            channel1 = (ts-i_bin/(self.num_bins-1)).unsqueeze(1)
            channel2 = pols.unsqueeze(1)
            channel3 = (xs/self.width).unsqueeze(1)
            channel4 = (ys/self.height).unsqueeze(1)
            input_data = torch.cat((channel1, channel2), dim=1)  
            input_data = torch.cat((input_data, channel3), dim=1)
            input_data = torch.cat((input_data, channel4), dim=1)
            values = ts * self.value_layer.forward(input_data)
            # draw in voxel grid
            idx = idx_before_bins + self.width * self.height * i_bin
            voxel_grid.put_(idx.long(), values, accumulate=True)

        voxel_grid = voxel_grid.view(1,self.num_bins, self.height, self.width)
        return voxel_grid
    
class CistaLSTCNet(nn.Module):
     def __init__(self, image_dim, base_channels=64, depth=5, num_bins=5):
          super(CistaLSTCNet, self).__init__()
          '''
               CISTA-LSTC network for events-to-video reconstruction
          '''
          self.num_bins = num_bins
          self.depth = depth
          self.height, self.width = image_dim
          self.num_states = 3
          
          self.event_repre_MLP = QuantizationLayer(num_bins = self.num_bins, height = self.height, width = self.width, mlp_layers=[2, 100, 100, 1], activation=nn.LeakyReLU(negative_slope=0.1))
          self.We = ConvLayer(in_channels=self.num_bins, out_channels=int(base_channels/2), kernel_size=3,\
          stride=1, padding=1) #We_new 
          self.Wi = ConvLayer(in_channels=1, out_channels=int(base_channels/2), kernel_size=3,\
               stride=1, padding=1) 
          self.W0 = ConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3,\
               stride=2, padding=1) 

          self.P0 = ConvLSTC(x_size=base_channels, z_size=2*base_channels, output_size=2*base_channels, kernel_size=3) 

          lista_block = IstaBlock(base_channels=base_channels, is_recurrent=False) 
          self.lista_blocks = nn.ModuleList([lista_block for i in range(self.depth)])
          

          self.Dg = RecurrentConvLayer(in_channels=2*base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1,
               activation='relu')

          self.upsamp_conv = UpsampleConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=0, activation='relu')
          
          self.final_conv = ConvLayer(in_channels=base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          
          self.sigmoid = nn.Sigmoid()


     def forward(self, events, prev_image, prev_states):
          '''
          Inputs:
               events: torch.tensor, float32, [batch_size, num_bins, H, W]
                    Event voxel grid
               prev_image: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame from the last reconstruction
               prev_states: None or list of torch.tensor, float32
                    Previous states
          Outputs:
               rec_I: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame
               states: list of torch.tensor, float32
                    Updated states in e2v_net
          '''
          
          if prev_states is None:
               prev_states = [None]*self.num_states
          states = [] 

          events = self.event_repre_MLP(events)
          x_E = self.We(events)
          x_I = self.Wi(prev_image)
          x1 = connect_cat(x_E, x_I) 

          x1 = self.W0(x1) 

          z, state = self.P0(x1, prev_states[-2], prev_states[0] if prev_states[0] is not None else None)
          states.append(state)
          tmp = z.clone()

          for i in range(self.depth):
               tmp = self.lista_blocks[i].D(tmp)
               x = x1- tmp
               x = self.lista_blocks[i].P(x)
               x = x + z  # + temporal_z
               z = softshrink(x, self.lista_blocks[i].Lambda) 
               tmp = z      

          states.append(z)
          
          rec_I, state = self.Dg(z, prev_states[-1])
          states.append(state)

          rec_I = self.upsamp_conv(rec_I)

          rec_I = self.final_conv(rec_I)
          rec_I = self.sigmoid(rec_I)

          return rec_I, states


class CistaTCNet(nn.Module):
     def __init__(self, base_channels=32, depth=5, num_bins=5):
          super(CistaTCNet, self).__init__()
          '''
               CISTA-TC network for events-to-video reconstruction
          '''
          self.num_bins = num_bins
          self.depth = depth
          self.num_states = 2 

          self.one_conv_for_prev = ConvLayer(in_channels=2*base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          self.one_conv_for_cur = ConvLayer(in_channels=2*base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          alpha = nn.Parameter(torch.Tensor([0.001*np.random.rand(2*base_channels, 1,1)]))
          self.alpha = nn.ParameterList([ alpha for i in range(self.depth)])

               
          self.We = ConvLayer(in_channels=self.num_bins, out_channels=int(base_channels/2), kernel_size=3,\
          stride=1, padding=1)
          self.Wi = ConvLayer(in_channels=1, out_channels=int(base_channels/2), kernel_size=3,\
               stride=1, padding=1) 
          self.W0 = ConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3,\
               stride=2, padding=1)

          self.P0 = ConvLayer(in_channels=base_channels, out_channels=2*base_channels, kernel_size=3,\
               stride=1, padding=1)#64

          lista_block = IstaBlock(base_channels=base_channels, is_recurrent=False)
          self.lista_blocks = nn.ModuleList([lista_block for i in range(self.depth)])

          self.Dg = RecurrentConvLayer(in_channels=2*base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1,
                    activation='relu') 

          self.upsamp_conv = UpsampleConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=0, activation=None, norm=None)
          self.final_conv = ConvLayer(in_channels=base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          
          self.sigmoid = nn.Sigmoid()
     def calc_attention_feature(self, img1, img2, prev_attention_state):# , prev_attention_state):
          # TSA
          S1 = self.sim_layers(img1)
          S2 = self.sim_layers(img2)
          feat1 =  self.one_conv1(S1)
          feat2 = self.one_conv2(S2)
          attention_map = torch.sigmoid(torch.mul(feat1,feat2)) #attention state
          # return attention_map
          if prev_attention_state is None:
               prev_attention_state = torch.ones_like(attention_map)
          attention1 = torch.mul(S1, prev_attention_state)
          attention2 = torch.mul(S2, attention_map)
          return attention1, attention2, attention_map

     def forward(self, events, prev_img, prev_states):
          '''
          Inputs:
               events: torch.tensor, float32, [batch_size, num_bins, H, W]
                    Event voxel grid
               prev_img: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame from the last reconstruction
               prev_states: None or list of torch.tensor, float32
                    Previous states
          Outputs:
               rec_I: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame
               states: list of torch.tensor, float32
                    Updated states in e2v_net
          '''
          # input event tensor Ek, Ik-1, Ik,
          if prev_states is None:
               prev_states = [None]*self.num_states
          states = [] 


          x_E = self.We(events)
          x_I = self.Wi(prev_img)

          x1 = self.W0(connect_cat(x_E, x_I) ) 
          z = self.P0(x1)
          tmp = z
          if prev_states[0] is None:
              prev_states[0] = torch.zeros_like(z)
          
          one_ch_prev_z = self.one_conv_for_prev(prev_states[0])
          for i in range(self.depth):
               one_ch_cur_z = self.one_conv_for_cur(tmp)
               attention_map = torch.sigmoid(torch.mul(one_ch_prev_z, one_ch_cur_z))
               temporal_z = attention_map*torch.mul((prev_states[0]-tmp), self.alpha[i])
               tmp = self.lista_blocks[i].D(tmp)
               x = x1- tmp
               x = self.lista_blocks[i].P(x)
               x = x + z + temporal_z
               z = softshrink(x, self.lista_blocks[i].Lambda) 
               tmp = z      

          states.append(z)
          
          rec_I, state = self.Dg(z, prev_states[-1])
          states.append(state)

          rec_I = self.upsamp_conv(rec_I)
          rec_I = self.final_conv(rec_I)
          rec_I = self.sigmoid(rec_I)
          
          return rec_I, states 

