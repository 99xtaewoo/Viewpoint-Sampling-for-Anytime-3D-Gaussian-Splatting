"""def to_MB(a):
    return a/1024.0/1024.0
    
print(f"After model to device: {to_MB(torch.cuda.memory_allocated()):.2f}MB")


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])


starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation 
starter.record()

# 측정하고 싶은 모델 프로세스  

ender.record()
torch.cuda.synchronize()
curr_time = starter.elapsed_time(ender)
print(curr_time)"""
 
np.random.normal(0, 2, 10)

