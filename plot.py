import matplotlib.pyplot as plt

psnr_interval_list = [0,0,0,0,0,0]
interval_list = [20,22,24,26,28,30]
with open('output.txt', 'r') as f:
    _l = f.readline()
    while _l:
        print(_l)
        if len(_l) > 5:
            psnr = float(str(_l[6:10]))
            print(psnr)
            if(psnr < 20):
                psnr_interval_list[0] += 1
            elif int((psnr-20)/2) > 0 and int((psnr-20)/2) < 6:
                psnr_interval_list[int((psnr-20)/2)] += 1
            else:
                psnr_interval_list[5] += 1
        _l = f.readline()


plt.plot(interval_list, psnr_interval_list)
plt.savefig("plt.png")