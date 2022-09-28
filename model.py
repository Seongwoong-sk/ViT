import torch
import torch.nn as nn

'''
    EX
    patch_size = 4, latent_vec_dim(D) = 32, layer 수 = 12, Patch 개수(N) = 64 * 64 / 4 * 4  = 256
    k,q 구할 때의 Dimension이 Dh = D / K :: 32/ 8 (e.g) = 4

    1. input 이미지 batch <100 x 3 x 64 x 64>
    2. 패치 : <100 x 256 x 48(P^2C)> --> P^2C인 벡터가 256개 있음.. 그게 100개 있음
    3. Linear Projection : <100 x 256 x 32>
    4. Class Token : <100 x 257 x 32>
    5. Positional Embedding 계산 : 4번하고 같은 크기를 더해주니깐 포함해서 사이즈가 이렇게 되고
    6. 4+5를 Transformer ENcoder에 input으로 넣음
    7. q,k,v = <100 x 257(N) x 32(D = Dh * K)>     // 단독으론 e.g    q = 100 x 257 x 4(Dh) x 8(K) --> 이렇게 되면 4차원으로 계산하니깐 3차원으로 계산하기 위해서 D만 언급 -> Dh * K를 담고 있음.
    8. 3개 모아져 있는 걸 각각 쪼개기 - view 함수 : <100 x 257 x 8 x 4> // head수에 따라서 q,k,v가 정해지는 거니깐 permute 함수를 이용해서 위치 바꿈 : <100 x 8 x 257 x 4> 
       -> 257 x 4 짜리 q,k,v라는 3개가 있는 녀석이 각 헤드마다 있음.  Attenton 할 때 transpose로 계산되니깐 k는 transpose해줌 < 100 x 257 x 4 x 8>
    9. 그래서 attention을 계산하게 되면 < 100 x 8 x 257 x 257 > 이 되고 v랑 이것을 곱해주는 거니깐 다시 연산이 되면 Av = <100 x 8 x 257 x 4>가 나옴.
    10. 그리고 reshape으로 concatenate시켜서  <100 x 8 x 257 x 4> --> <100 x 257 x 32> (이게 Multi-Head Attention의 결과 사이즈)
    11. 그리고 MLP를 거치면 똑같이 나옴 -> <100 x 257 x 32> :: nn.Linear(latent_vec_dim)의 input과 output이 동일한 사이즈 /  이 과정이 12번 돌았다고 가정을 하면 MLP에 넣어서 < 100 x 10>이 됨
       -> 이미지 100개의 각각의 클래스가 10개짜리 벡터들로 나옴.
'''


class LinearProjection(nn.Module):


    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        '''
        linear_proj
        :: patch_data를 linear 한번 함, 1 x P^2C짜리를 1 x D로 변환해주는 것
        :: 그래서 input이 P^2C가 되고, output이 D가 됨.
        :: linear projection 후 token 붙여줘야 함 

        cls_token
        :: 1 x D 짜리를 가장 왼쪽에다가 concatenate 시켜주는 것이니깐 1 x D 짜리 텐서를 만들고, 얘는 학습 가능해야 되니깐 파라미터로 정의
        :: nn.Parameter 라고 함은 모델 업데이트를 할 때 같이 업데이트 되는 변수를 의미
        
        pos_embedding
        :: positional embedding도 학습 가능해야되니깐 Parameter로 정의
        :: 얘는 class token을 붙였기 때문에 patch 수에 +1이 됨. 
        :: 그 다음 latent_vec와 똑같아서 D가 됨. --> 1 x N+1 x D

        '''
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim)
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vec_dim))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0) # 실제 데이터는 batch 사이즈로 들어옴 /  B(batch_size) x M x P^2C로 들어와서

        # linear projection을 해주면 B x N x D가 됨 / 그리고 class token은 1 x D니깐 각각의 batch에 동일한 cls token을 넣어줘야 되니깐 repeat함수를 써서 B x 1 x D 를 만든 다음에
        # self.linear_proj한 결과와 concatenate시킴.
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1) 
        x += self.pos_embedding # 그 다음 x에다가 동일한 크기를 가진 positional embedding을 더해주고, 
        x = self.dropout(x) # 마지막 나오는 결과에 정규화를 위해 dropout 적용
        return x
# 이렇게 Linear Projection을 거치면 Transformer의 Input이 완성이 됨.



class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        '''
        q,k,v 계산을 하는 데, linear였음
        query: 들어온 input값 x *  W의 q  -> Linear 연산
        '''
        
        # query의 Dimension은 Dh가 되어야 함. / nn.Linear(D, Dh)  :: D를 Dh로 넘겨주는 것 //  Dh : D / k  //  D : K * Dh   (Dh를 k(head 수)만큼 계산했다는 뜻)
        # 실제 코딩을 할 때 이런 식으로 계산하게 되면 Multi-Head가 한 번에 연산이 됨.
        # latent_vec_dim이 가장 중요한 부분 :: 어떻게 하면 Multi-Head Attention을 가장 간단하게 표현할 수 있을 지

        self.query = nn.Linear(latent_vec_dim, latent_vec_dim)  # 모든 head의 query 구함
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim) # 모든 head의 key 구함
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim) # 모든 head의 value 구함
        # 마지막 변수로 있는 latent_vec_dim은 D로써 K * Dh니깐 K이랑 Dh랑 나눠줘야지 Head마다의 각 쿼리,키,밸류가 나옴  ###

        self.scale = torch.sqrt(self.head_dim*torch.ones(1)).to(device) # 학습이 되면 안되니깐 tensor로만 정의를 하는 데, tensor 하나만 만들면 CPU 연산만 되는 텐서 -> GPU 연산이 될 수 있도록 to(device)
        self.dropout = nn.Dropout(drop_rate)



    def forward(self, x): # Linear Projection에서 나온 결과가 input으로 들어옴. (LayerNorm 한 번 거치고)
        batch_size = x.size(0)
        # k,q,v를 계산한 다음에
        q = self.query(x) 
        k = self.key(x)
        v = self.value(x)

        # Multi-Head Attention 계산
        # ### 이 부분에서 재정비
        '''
        Batch size가 있고, N (vector의 개수), head의 개수, head의 dimension
        head의 개수 (self.num_heads)와 head의 dimension(self.head_dim)을 곱하면 latent_vec_dim
        -> 그래서 latent_vec_dim을 num_heads와 head_dim으로 나눴단 얘기
        그 다음 head 수를 앞으로 보내기 위해서 permute을 이용해 vector의 개수와 heads 수 위치 바꿔줌
        '''
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # 현재 이게 k.Transpose // k 같은 경우는 나중에 q * k.T로 유사도를 계산하기 때문에 1,3 - 3,1 이렇게 세팅
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) # value도 마찬가지로 q랑 동일한 크기가 나와야되니깐 똑같이 재정비
        # q,k,v 의 모든 head에 대한 값을 구했다면 

        # softmax 값을 취해서 계산을 해주면 됨.
        # -> softmax 값 취할 때 q * k.T 해준 것 (matrix 곱은 @으로도 처리 가능.  @ = torch.matmul)
        # -> 마지막에 scale 나눠줘야함. key dimension에 root 씌운 걸로 나눔 -> 출력 값이 커짐에 따라서 Softmax 때 문제가 발생해서 Scale 사용
        attention = torch.softmax(q @ k / self.scale, dim=-1) # softmax 하면 attention이 나오게 됨. 논문에선 A  
        x = self.dropout(attention) @ v # A와 v를 matrix 곱
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim)

        return x, attention

# Transformer의 Layer 하나에 해당
# Multi-Head Attention과 MLP와 LayerNorm으로 구성
# 

class TFencoderLayer(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(latent_vec_dim=latent_vec_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate) # 정규화를 위해 사용
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim), # MLP는 층이 2개에다가 GELU 사용
                                 nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(drop_rate))

    def forward(self, x):
        z = self.ln1(x) # 처음에 layer norm  해주고
        z, att = self.msa(z) # MHA 구하고 
        z = self.dropout(z)
        x = x + z # 이전 값 더하고
        z = self.ln2(x) # layer norm 하고
        z = self.mlp(z) # MLP 구하고
        x = x + z # 이전 값 더하고

        return x, att


# ViT 전체 구조
class VisionTransformer(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes):
        super().__init__()

        '''
        LinearProjection
        patch_vec_size : p^2C
        num_patches : patch의 개수 ,  hw/p^2
        latent_vec_dim = D
        drop_rate = dropout rate

        LP가 끝나면 Class Token 넣고 Patch Embedding 과정이 있는데 이것도 포함돼있음.
        '''
        self.patchembedding = LinearProjection(patch_vec_size=patch_vec_size, num_patches=num_patches,
                                               latent_vec_dim=latent_vec_dim, drop_rate=drop_rate)

        '''
        self.patchembedding이 끝나면 transformer에 넣어줘야함.
        transformer가 반복디 되기 때문에 그 반복을 list안에 넣음. -> list 안에 for문을 사용하게 되면 TFencoderLayer라는 클래스를 num_layers 만큼 반복하여 append하게 됨.
        각각의 클래스를 12번 선언한 것이기 때문에 각 layer는 파라미터를 공유하지 않고 독립적인 파라미터 사용.
        이렇게 만들어진 list를 ModuleList에 넣어서 학습에 사용할 수 있도록 해줌
        transformer의 layer들이 쭉 저장이 되어있고, 마지막에 transformer encoder로부터 나온 결괏값의 첫번째 class token에 해당되는 그 vector만 뽑아서 분류를 할 것임.
        '''
        self.transformer = nn.ModuleList([TFencoderLayer(latent_vec_dim=latent_vec_dim, num_heads=num_heads,
                                                         mlp_hidden_dim=mlp_hidden_dim, drop_rate=drop_rate)
                                          for _ in range(num_layers)]) # 12

        '''
        self.mlp_head
        그래서 mlp_head 부분에선 latent_vec_dim 하나짜리를 넣어서 크기가 하나가 됨.
        Linear 넣기 전에 LayerNorm 한 번 해주고, 그 다음 그 벡터 크기 D짜리가 Linear에 들어와서 class의 개수와 동일하게 노드를 가진 output을 출력하게 됨. 
        '''
        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim), nn.Linear(latent_vec_dim, num_classes))


    def forward(self, x):
        att_list = [] # attention list : 학습이나 평가할 땐 필요 없는데, attention 값을 저장할려고 만듦
        x = self.patchembedding(x) # patch embedding해서 positional embedding까지 완료한 녀석 -> transformer의 실질적인 input
        for layer in self.transformer: # 그 input을 , transformer가 리스트로 쌓여 있는데, 쌓여있는 리스트에 대해서 하나씩 for문을 이용해서 하나씩 불러옴
            x, att = layer(x) 
            att_list.append(att) # 각 층마다 나오는 attention을 리스트에 저장
        x = self.mlp_head(x[:,0]) # layer를 다 거치고 난 다음에 가장 앞 부분 (class token에 해당되는 부분)의 vector만 떼다가 mlp_head에 넣음

        return x, att_list # 최종 결괏값 산출, attention 모아놓은 list 산출
