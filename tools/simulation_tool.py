import random


def run_simulation(noise):
    """
    模拟TVB simulation
    这里只是先用假数据测试Agent结构
    后面替换成真实TVB
    """

    # 假设noise越大，频率越低
    freq = 1.5 - noise * 1000

    # 加一点随机扰动
    freq += random.uniform(-0.05, 0.05)

    return freq