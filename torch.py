import torch


print("\nCUDA is avaiable:{}, version is {}".
      format(torch.cuda.is_available(), torch.version.cuda))

# 返回gpu数量
print(torch.cuda.device_count())

# 返回gpu名字，设备索引默认从0开始
print(torch.cuda.get_device_name(0))

# 返回当前设备索引
print(torch.cuda.current_device())