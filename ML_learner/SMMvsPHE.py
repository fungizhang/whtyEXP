from phe import paillier # 开源库
import time # 做性能测试
import numpy as np

import numpy as np
import math
import shamirs

# a = shamirs.shares(5, 2)
# b = shamirs.interpolate(a)

np.random.seed(1234)

### parameters
user_num = 100000
bucket_size = 5

### generate Ap's g and h, PP's feature value of x
D_AP_g = np.random.randn(user_num).tolist()
D_AP_h = np.random.randn(user_num).tolist()
D_PP_x = np.random.randn(user_num).tolist()

### get the sorted feature value of x
D_PP_sorted = np.sort(D_PP_x)


### compute the number of columns of mat_PP
bucket_num = math.ceil(user_num / bucket_size) - 1

### initialize mat_PP
mat_PP = np.array(np.zeros((user_num, bucket_num)))



print("=======================================SMM")

a = time.time()
### generate mat_PP
bucket_id = 0
for i in range(user_num):
    if i == 0:
        pass
    elif i % bucket_size == 0:
        threshold_1 = D_PP_sorted[i]
        threshold_2 = D_PP_sorted[i-1]
        threshold = (threshold_1 + threshold_2) / 2

        # reverse 0 to 1 if the feature value is not larger than threshold
        for j in range(user_num):
            if D_PP_x[j] < threshold:
                mat_PP[j][bucket_id] = 1

        # next bucket_id
        bucket_id += 1

b = time.time()
print("构建PP端矩阵用时：", b-a)


a = time.time()
### generate mat_AP
mat_AP = np.array(np.zeros((user_num, 2)))
for i in range(user_num):
    mat_AP[i][0] = D_AP_g[i]
    mat_AP[i][1] = D_AP_h[i]

b = time.time()
print("构建AP端矩阵用时：", b-a)

# aaa = np.matrix(mat_AP)

a = time.time()
### compute Q
q, r = np.linalg.qr(mat_AP, mode='complete')

b = time.time()
print("QR分解用时：", b-a)

### get Z. note that we just test the time-consuming of SMM. So, we set the entire Q2 as Z.
Z = q[:,2:102]


a = time.time()
### compute W
ZT = Z.T
I = np.eye(user_num)
c = time.time()
ZZT = Z.dot(ZT)
d = time.time()
print("ZZT用时：", d-c)

c = time.time()
tmp = I - ZZT
d = time.time()
print("减法用时：", d-c)

c = time.time()
W = tmp.dot(mat_PP)
d = time.time()
print("(I - ZZT)mat_PP用时：", d-c)

c = time.time()
### compute mat_PA * mat_PP
mat_AP_PP = mat_AP.T.dot(W)
d = time.time()
print("mat_AP.T.dot(W)用时：", d-c)

b = time.time()
print("矩阵相乘用时：", b-a)

### compared with No SMM
a = time.time()
res = mat_AP.T.dot(mat_PP)

b = time.time()
print("明文相乘用时：", b-a)


print("=======================================PHE")


# PHE
print("默认私钥大小：",paillier.DEFAULT_KEYSIZE) #3072
# generate public and private key
public_key,private_key = paillier.generate_paillier_keypair()

# encryption
a = time.time()
encrypted_message_list_g = [public_key.encrypt(m) for m in D_AP_g]
encrypted_message_list_h = [public_key.encrypt(m) for m in D_AP_h]
b = time.time()
print("加密耗时s：",b-a)


# compute the aggregated g and h according to different split nodes
a = time.time()
agg_g_list = []
agg_h_list = []
for i in range(user_num):
    if i == 0:
        pass
    elif i % bucket_size == 0:
        threshold_1 = D_PP_sorted[i]
        threshold_2 = D_PP_sorted[i-1]
        threshold = (threshold_1 + threshold_2) / 2

        #
        agg_g = 0
        agg_h = 0
        for j in range(user_num):
            if D_PP_x[j] < threshold:
                agg_g += encrypted_message_list_g[j]
                agg_h += encrypted_message_list_h[j]

        agg_g_list.append(agg_g)
        agg_h_list.append(agg_h)
        # next bucket_id
        bucket_id += 1

b = time.time()
print("计算聚合g/h的用时：",b-a)

# decryption
a = time.time()
decrypted_message_list_g = [private_key.decrypt(c) for c in agg_g_list]
decrypted_message_list_h = [private_key.decrypt(c) for c in agg_h_list]
b = time.time()
print("解密用时：",b-a)


# show three results

print("安全矩阵相乘的结果", mat_AP_PP)
print("加密运算的结果", [decrypted_message_list_g, decrypted_message_list_g])
print("明文结果", res)
