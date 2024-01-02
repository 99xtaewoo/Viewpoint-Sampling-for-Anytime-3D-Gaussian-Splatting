

"""

1, 2, 3, 4, 5, 6 ,7 , 8, 9 ,10 이 있어. 
step =2 이면,  1 -> 3 -> 5 -> 7 -> 9  총 4번 반복하고,
시작점을 한 칸 옮겨, 2->4->6->8->10 후 ,
또 한칸 시작점을 옮기는데, 3 -> 5인데, 이미 과거에 구했던 값이잖아?
즉, step 값과 시작점 값이 같으니까 또 반복하지 않아. 
그래서, 이번에는 step size를 1을 증가하고, 
start 를 0으로 초기화하고 1 -> 4 -> 7 -> 10 이 되는거야. 
즉, step size와 start 가 같을때 , start 를 0으로 초기화하는거야 

정리하면,
반복할 때마다, 시작점을 한 칸씩 옮긴다.
step 이 시작점과 같아지면 , 시작점을 처음으로 옮기고, step 을 1을 증가시켜
반복한다.
이럴때, N = 219 이고 step 이 8에서 13까지 증가할때 총 반복횟수는 얼마인가?




알고리즘 설명

시작점을 한 칸씩 옮긴다.
step 이 시작점과 같아지면 , 시작점을 처음으로 옮기고, step 을 1을 증가시켜
반복한다.
이럴때, N = 219 이고 step 이 8에서 13까지 증가할때 총 반복횟수는 얼마인가?

"""
def find_iteration(N,step,stop):
    start = 0
    step = step
    iter = 0
    viewpoint_stack = None
    while True:
        
        if not viewpoint_stack:
            viewpoint_stack = [1] * N
            if step > stop - 1:
                answer = iter
                print(answer)
                break
            if start == step:
                start = 0
                step += 1

            viewpoint_stack = viewpoint_stack[start::step]
            print("step :" ,step) 
            start += 1  
        iter += 1
        viewpoint_stack.pop(0)

    return answer


print(find_iteration(219,8,13))
