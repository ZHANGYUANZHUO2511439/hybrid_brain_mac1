from agents.so_agent import SOAgent
from tools.simulation_tool import run_simulation
from tools.evaluation_tool import is_slow_oscillation


def main():

    # 创建Agent
    agent = SOAgent()

    # 初始noise
    noise = 0.0001

    print("========== SO AGENT START ==========\n")

    for step in range(30):

        # 运行simulation
        freq = run_simulation(noise)

        print(f"Step {step}")
        print(f"Current noise: {noise:.6f}")
        print(f"Current frequency: {freq:.3f} Hz")

        # 判断是否进入SO
        if is_slow_oscillation(freq):

            print("\nSO reached!")
            print("Agent stopping.")

            break

        # Agent更新noise
        new_noise = agent.update_noise(noise, freq)

        print(f"Updated noise: {new_noise:.6f}")
        print("-----------------------------------\n")

        noise = new_noise

    else:

        print("Max iteration reached.")


if __name__ == "__main__":

    main()