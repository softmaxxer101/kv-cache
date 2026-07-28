import torch, torch.nn as nn, torch.nn.functional as F



D = 512

h = 16

hd = D // h



wq, wk, wv, wo = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)



# the cache: just two tensors

cache_k = None

cache_v = None



def forward_step(x):

    global cache_k, cache_v

    B, T, D = x.shape  # T = 1 during generation (one new token)



    q, k, v = wq(x), wk(x), wv(x)

    q = q.reshape(B, T, h, hd).transpose(1, 2)

    k = k.reshape(B, T, h, hd).transpose(1, 2)

    v = v.reshape(B, T, h, hd).transpose(1, 2)



    # --- this is the whole "cache" ---

    if cache_k is None:

        cache_k, cache_v = k, v

    else:

        cache_k = torch.cat([cache_k, k], dim=2)  # append along seq-len dim

        cache_v = torch.cat([cache_v, v], dim=2)

    # k, v = cache_k, cache_v

    # ----------------------------------



    attention = F.softmax((q @ cache_k.transpose(-2, -1)) / (D ** 0.5), dim=-1) @ cache_v

    out = wo(attention.transpose(1, 2).reshape(B, T, D))

    return out



x1 = torch.randn(1, 1, D)   # token 1 = 'a'

out1 = forward_step(x1)

print(f"cache_k shape : {cache_k.shape} | cache_v shape : {cache_v.shape}")   # cache_kv has 1 token

print("----------")



x2 = torch.randn(1, 1, D)   # token 2 = 'cat'

out2 = forward_step(x2)

print(f"cache_k shape : {cache_k.shape} | cache_v shape : {cache_v.shape}")   # cache_kv has 2 token

print("----------")



x3 = torch.randn(1, 1, D)   # token 3 = 'sat'

out4 = forward_step(x1)

print(f"cache_k shape : {cache_k.shape} | cache_v shape : {cache_v.shape}")   # cache_kv has 3 token

print("----------")



x4 = torch.randn(1, 1, D)   # token 4 = 'on'

out4 = forward_step(x1)

print(f"cache_k shape : {cache_k.shape} | cache_v shape : {cache_v.shape}")   # cache_kv has 4 token

print("----------")



x5 = torch.randn(1, 1, D)   # token 5 = 'the'

out5 = forward_step(x1)

print(f"cache_k shape : {cache_k.shape} | cache_v shape : {cache_v.shape}")   # cache_kv has 5 token

print("----------")



x6 = torch.randn(1, 1, D)   # token 6 = 'mat'

out6 = forward_step(x1)

print(f"cache_k shape : {cache_k.shape} | cache_v shape : {cache_v.shape}")   # cache_kv has 6 token

print("----------")
