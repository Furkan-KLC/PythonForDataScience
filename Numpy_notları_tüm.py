# %%
import numpy as np



# %%
arr=np.array([2,55,34,8,11,17,60,94,100])

# %% [markdown]
# ARRAY MEMORY 

# %%
print(arr. flags["WRITEABLE"] )  # means array owns its memory 
print(arr.flags["C_CONTIGUOUS"])
print(arr.flags["OWNDATA"])

# %%
arr.shape

# %%
arr.strides             #Tuple of bytes to step in each dimension when traversing an array.

# %%
print(arr.ndim)         #Number of array dimensions.
arr.size
arr.itemsize
arr.nbytes              #Total bytes consumed by the elements of the array.

arr.base                #Base object if memory is from some other object.
y=arr[1:4]
y.base is arr

# %%
arr.dtype      	  #Data-type of the array's elements

# %%
complex_arr=np.array([ 381.          +0.j        ,   82.65246204+135.09934947j,
       -103.91594351 +15.94751083j,  -85.5         -7.79422863j,
        -74.73651853 +15.94812435j,  -74.73651853 -15.94812435j,
        -85.5         +7.79422863j, -103.91594351 -15.94751083j,
         82.65246204-135.09934947j])
real_part=complex_arr.real
real_part
imaginary_part=complex_arr.imag
imaginary_part


# %%
new_arr=arr.reshape(3,3)
new_arr
print(new_arr.T)   #TRANSPOSE OF AN ARRAY.

# %% [markdown]
# ------ARRAY METHODS------
# 

# %%
arr.item(2)   #0 dan başlayarak 2. elemanı seçer
arr.dtype

# %%
my_arr=arr.astype(str,casting='safe',copy=False)   #veri tipini degistirdik normalde int di
my_arr.dtype

# %%
arr.getfield(float) #Veriyi ham bitleriyle tutar ama başka tipmiş gibi okur

# %%
arrayim=np.random.random((3,2))
arrayim

# %%
zeros=np.zeros((4,2))
zeros

# %%
empty_arr=np.empty(4)
empty_arr.fill(6)         #verilen sayı ile arrayi doldurur
empty_arr

# %% [markdown]
# -------SHAPE MANUPILATION ------

# %%
arr=arr.reshape(3,3)

# %%
empty_arr.resize(3,2,refcheck=False)      #yeniden boyutlandırır ve ekleme yapar.
empty_arr

# %%
arr.swapaxes(0,1)    #axis 0 (satırlar) ↔ axis 1 (sütunlar) 


# %%
two_dim=np.random.randint(10,80,20).reshape(4,5)   #two dimension to one dimension
two_dim
flat=two_dim.flatten()
flat

# %%
dummy_array = np.array([[[42], [51], [63]]])  # shape: (1, 3, 1)        
print("Orijinal şekil:", dummy_array.shape)

squeezed = np.squeeze(dummy_array)               #3 boyutlu arrayi tek boyutlu hale getirdi
print("Sıkıştırılmış şekil:", squeezed.shape)
print(squeezed)

# %% [markdown]
# --------SELECTION AND MANUPILATION-------------:

# %%
two_dim

# %%
two_dim[1:3]                             #1 satırdan 3 . satıra kadar verir 3 dahil değil

# %%
two_dim[1:3,0:2]                     #1 ve 2. satırın 0 ve 1. elemanları    [satır:satır,sütun:sütun]

# %%
two_dim.take([1,2,3])                   #verilen indislerdekini seçer 

# %%
two_dim.put(1,11)                      #verilen indise verilen numarayı yazar
two_dim.take([1,2,3])  

# %%
two_dim.repeat(3)                      #her elemanı verilen sayı kadar tekrar  yazar

# %%
two_dim.sort()
two_dim

# %%
two_dim.argsort()              #elemanların kaçıncı büyük olduğunu syöler biz zaten sıralamıştık

# %%
partitioned=two_dim.ravel()                       
                                       
partitioned.partition(8)    #the value of the element in k-th position is in the position it would be in a sorted array. In the output array,
partitioned                  #verilen indeksteki elemanı ortaya küçüklerini sola büyüklerini sağa yazar bu örenkte 8. indekste 33 vardı




# %%
two_dim.nonzero()            #0 olmayan her satırdaki eleman için  o satırın indeks numarasını yazar. ve 2. arrayda her saatır için 0 dan n e kadar yazar.

# %%
arr = np.array([10, 20, 30, 40, 50])                 #condition: Boolean (True/False) değerler içeren bir 1D dizi.
                                                       #axis: Filtrenin uygulanacağı eksen (0: satır, 1: sütun).
mask = [True, False, True, False, True]
filtered = np.compress(mask, arr)                      #array: Filtrelenecek NumPy dizisi.
print(filtered)  # [10 30 50]


# %% [markdown]
# ----------CALCULATION------------

# %%
arr

# %%
arr.max()

# %%
arr.min()

# %%
arr.argmin()

# %%
arr.clip(20,50)          #alt ve üst değer değişikliği yapar 20 den küçükleri 20'ye büyükleri de 50 ye eşitler

# %%
comp= np.array([1+2j, 3-4j, -2+0.5j, 0-1j])
comp.conj()                                                 #complex conjugate ini verir

# %%
fractions=np.array([1.234, 5.67891, 0.0001234, -2.3456])
fractions.round(2)                                          #küsürat kısmını verilen basmağa indirir 

# %%
arr=np.random.randint(10,90,16).reshape(4,4)
arr

# %%
arr=np.random.randint(10,90,16).reshape(4,4)
arr.sum(axis=0)    #satırları toplar verir
arr.sum(axis=1)    #sütunları toplar verir 

# %%
prices = np.array([100, 102, 105, 107, 110])

# %%
np.diff(prices)

# %% [markdown]
# .sin .cos. arctan .arcos .hpot .degrees. radians .round. .florr .ceil .sum .cumsum .log .exp .log10 .power .pow .mod .sqrt .absolute GİBİ PEK ÇOK MATH FONKSİYONU MEVCUT 

# %%
arr.cumsum(axis=0)    #kümülatif toplamı verir

# %%
darray=np.array([10,22,5,32,84,55])
darray.var()      #arrayin varyansını verir 

# %%
darray.std()     #standart sapmasını verir

# %%
darray.prod(axis=0)   #verilen eksende elemanların çarpımı

# %%
darray.all()    #tüm değerler tru ise true döndürür     
darray.any()     #1 tane bile true vars true verir 

# %% [markdown]
# FOURIER TRANSFORM AND INVERSE FOURIER

# %%
fft_arr=np.fft.fft(arr)   #Compute the one-dimensional discrete Fourier Transform.
fft_arr

# %%
normal_arr=np.fft.ifft(fft_arr).real   
normal_arr


# %% [markdown]
# LINEAR ALGEBRA

# %%
first_arr=np.array([58,3,7,99])
second_arr=np.array([22,1,43,11])
result=np.dot(first_arr,second_arr)
print(result)


# %%
result=np.vdot(first_arr,second_arr)
result

# %%
first_arr=first_arr.reshape(2,2)
second_arr=second_arr.reshape(2,2)
result1=np.linalg.eig(first_arr)         #eigen value of matrix
result1

# %%
det=np.linalg.det(second_arr)   #determinant of array
det

# %%
inverse=np.linalg.inv(first_arr)
inverse

# %%
t=np.linalg.matrix_transpose(first_arr)
t

# %%
a = np.array([[1, 2], [3, 5]])   # 1*x0 + 2 * x1 = 1    and   3 * x0 + 5 * x1 = 2
b = np.array([1, 2])             # a arrayi katsayılar arrayi b arrayi sonuç arrayi solve methodu ile çözüyor
x = np.linalg.solve(a, b)


