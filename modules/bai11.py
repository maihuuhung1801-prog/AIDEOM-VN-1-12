"""
modules/bai11.py
Bài 11 — Học Tăng Cường (Q-learning) cho Chính sách Kinh tế Thích nghi
Chuyển đổi từ Jupyter Notebook sang Streamlit module.
Giữ nguyên toàn bộ logic môi trường, thuật toán, MDP.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")
import streamlit as st
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# 1. MÔI TRƯỜNG VietnamEconomyEnv
# ──────────────────────────────────────────────────────────────────
def load_data():
    """Trả về cấu hình hằng số và class môi trường."""
    ACTION_NAMES = [
        "a0: Truyền thống",
        "a1: Cân bằng",
        "a2: Số hóa nhanh",
        "a3: AI dẫn dắt",
        "a4: Bao trùm",
    ]
    ALLOCATION = {
        0: np.array([0.70, 0.10, 0.10, 0.10]),
        1: np.array([0.40, 0.25, 0.15, 0.20]),
        2: np.array([0.25, 0.45, 0.15, 0.15]),
        3: np.array([0.20, 0.20, 0.45, 0.15]),
        4: np.array([0.30, 0.20, 0.10, 0.40]),
    }
    return ACTION_NAMES, ALLOCATION


class VietnamEconomyEnv:
    """
    Môi trường mô phỏng kinh tế Việt Nam (MDP đơn giản hóa).
    State: (gdp_level, digital_level, ai_level, unemp_level) ∈ {0,1,2}^4 → 81 trạng thái.
    Action: 5 chiến lược phân bổ ngân sách [K, D, AI, H].
    Episode: T = 10 năm.
    Reward: 0.40·ΔGDP − 0.25·ΔU − 0.20·CyberRisk − 0.15·Emission
    """

    ACTION_NAMES = [
        "a0: Truyền thống",
        "a1: Cân bằng",
        "a2: Số hóa nhanh",
        "a3: AI dẫn dắt",
        "a4: Bao trùm",
    ]
    STATE_LABELS = ["GDP growth", "Digital index", "AI capacity", "Unemp risk"]
    LEVEL_LABELS = ["low", "medium", "high"]

    def __init__(self):
        self.n_actions = 5
        self.T = 10
        self.allocation = {
            0: np.array([0.70, 0.10, 0.10, 0.10]),
            1: np.array([0.40, 0.25, 0.15, 0.20]),
            2: np.array([0.25, 0.45, 0.15, 0.15]),
            3: np.array([0.20, 0.20, 0.45, 0.15]),
            4: np.array([0.30, 0.20, 0.10, 0.40]),
        }
        self.w = np.array([0.40, 0.25, 0.20, 0.15])
        self._rng = np.random.default_rng(42)

    def reset(self, init_state=None):
        if init_state is None:
            init_state = np.array([1, 1, 0, 1])
        self.state   = init_state.copy()
        self.t       = 0
        self.K       = 27500.0
        self.D       = 20.3
        self.AI      = 86.0
        self.H       = 30.0
        self.L       = 54.0
        self.Y_prev  = None
        return self.state.copy()

    def step(self, action):
        a      = self.allocation[action]
        budget = 1000.0
        self.K  += a[0] * budget
        self.D  += a[1] * budget / 100.0
        self.AI += a[2] * budget / 20.0
        self.H  += a[3] * budget / 200.0

        Y = (self.K**0.33 * self.L**0.42
             * self.D**0.10 * self.AI**0.08 * self.H**0.07)
        delta_Y = (Y - self.Y_prev) / self.Y_prev if self.Y_prev else 0.0
        self.Y_prev = Y

        delta_GDP   =  delta_Y * 10
        delta_unemp = -a[0] * 0.05
        cyber_risk  =  max(0, a[2] - 0.25)
        emission    =  a[0] * 0.3
        reward = (self.w[0]*delta_GDP
                  - self.w[1]*abs(delta_unemp)
                  - self.w[2]*cyber_risk
                  - self.w[3]*emission)

        gdp_lvl   = self._to_level(delta_Y,    [-0.02, 0.05],  [0, 1, 2])
        dig_lvl   = self._to_level(self.D,     [20, 25],       [0, 1, 2])
        ai_lvl    = self._to_level(self.AI,    [100, 150],     [0, 1, 2])
        unemp_lvl = self._to_level(-a[1]*0.5-a[3]*0.5, [-0.2, 0.0], [2, 1, 0])
        self.state = np.array([gdp_lvl, dig_lvl, ai_lvl, unemp_lvl])

        self.t += 1
        done = self.t >= self.T
        return self.state.copy(), reward, done

    def _to_level(self, val, thresholds, levels):
        if val < thresholds[0]:
            return levels[0]
        elif val < thresholds[1]:
            return levels[1]
        else:
            return levels[2]

    def sample_action(self):
        return int(self._rng.integers(0, self.n_actions))

    def state_desc(self, s):
        return (f"GDP={self.LEVEL_LABELS[s[0]]}, "
                f"D={self.LEVEL_LABELS[s[1]]}, "
                f"AI={self.LEVEL_LABELS[s[2]]}, "
                f"U={self.LEVEL_LABELS[s[3]]}")


# ──────────────────────────────────────────────────────────────────
# 2. Q-LEARNING
# ──────────────────────────────────────────────────────────────────
def run_model(n_episodes=10000, alpha=0.1, gamma=0.95,
              eps_start=1.0, eps_end=0.05):
    """
    Câu 11.3.2 — Q-learning Tabular.
    Bellman update: Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') − Q(s,a)]
    """
    env = VietnamEconomyEnv()
    Q   = np.zeros((3, 3, 3, 3, 5), dtype=np.float64)
    episode_rewards = []
    episode_eps     = []

    np.random.seed(42)
    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (n_episodes * 0.8))
        s = env.reset()
        ep_reward = 0.0
        while True:
            if np.random.rand() < eps:
                a = env.sample_action()
            else:
                a = int(np.argmax(Q[tuple(s)]))
            s2, r, done = env.step(a)
            ep_reward += r
            Q[tuple(s) + (a,)] += alpha * (
                r + gamma * Q[tuple(s2)].max() - Q[tuple(s) + (a,)]
            )
            s = s2
            if done:
                break
        episode_rewards.append(ep_reward)
        episode_eps.append(eps)

    return Q, episode_rewards, episode_eps


# ──────────────────────────────────────────────────────────────────
# 3. ĐÁNH GIÁ CHÍNH SÁCH
# ──────────────────────────────────────────────────────────────────
def evaluate_policy(policy_fn, n_eval=500, seed=0):
    """Đánh giá chính sách qua n_eval episodes, trả về list reward."""
    env = VietnamEconomyEnv()
    rewards = []
    np.random.seed(seed)
    for _ in range(n_eval):
        s = env.reset()
        total_r = 0.0
        while True:
            a = policy_fn(s)
            s, r, done = env.step(a)
            total_r += r
            if done:
                break
        rewards.append(total_r)
    return rewards


def get_policy(Q, s):
    """π*(s) = argmax_a Q(s,a)."""
    q_vals = Q[tuple(s)]
    return int(np.argmax(q_vals)), q_vals


# ──────────────────────────────────────────────────────────────────
# 4. BIỂU ĐỒ
# ──────────────────────────────────────────────────────────────────
def chart_learning_curve(episode_rewards, episode_eps, n_episodes):
    """Câu 11.3.2 — Learning curve + epsilon decay."""
    window   = 200
    smoothed = pd.Series(episode_rewards).rolling(window, min_periods=1).mean()
    eps_end  = episode_eps[-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(range(n_episodes), episode_rewards,
            alpha=0.15, color="#90CAF9", lw=0.5, label="Raw")
    ax.plot(range(n_episodes), smoothed,
            color="#1565C0", lw=2, label=f"Smoothed ({window} ep)")
    ax.axvspan(0, int(n_episodes*0.4), alpha=0.05, color="red")
    ax.axvspan(int(n_episodes*0.8), n_episodes, alpha=0.05, color="green")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tổng Reward / Episode")
    ax.set_title("Learning Curve — Q-learning", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(n_episodes), episode_eps, color="#E91E63", lw=2)
    ax.fill_between(range(n_episodes), eps_end, episode_eps,
                    alpha=0.15, color="#E91E63")
    ax.axhline(eps_end, color="gray", linestyle="--", lw=1.5,
               label=f"ε_min = {eps_end}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε (epsilon)")
    ax.set_title("Epsilon Decay: Khám phá → Khai thác", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(n_episodes*0.1, 0.85, "Khám phá nhiều\n(random)", fontsize=9, color="red")
    ax.text(n_episodes*0.75, 0.15, "Khai thác\nQ-table",     fontsize=9, color="green")

    plt.tight_layout()
    return fig


def chart_policy_heatmap(Q):
    """Câu 11.3.3 — 4 heatmap chính sách."""
    env = VietnamEconomyEnv()
    action_cmap = plt.colormaps["Set1"].resampled(5)
    labels_short = ["Trthg", "Cân bằng", "Số hóa", "AI-DT", "Bao trùm"]
    LEVEL_L      = ["Low", "Mid", "High"]

    configs = [
        {"fix": {"ai":0,"u":1}, "x":"GDP", "y":"Digital", "title":"AI=low, U=med"},
        {"fix": {"ai":2,"u":0}, "x":"GDP", "y":"Digital", "title":"AI=high, U=low"},
        {"fix": {"d":1,"u":1},  "x":"GDP", "y":"AI",      "title":"D=med, U=med"},
        {"fix": {"g":1,"d":1},  "x":"AI",  "y":"Unemp",   "title":"GDP=med, D=med"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Heatmap Chính sách Tối ưu π*(s)\n"
        "(Màu = hành động | 0=Truyền thống → 4=Bao trùm)",
        fontsize=12, fontweight="bold"
    )

    for ax, cfg in zip(axes.flat, configs):
        pg = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                if cfg["x"] == "GDP" and cfg["y"] == "Digital":
                    s = np.array([i, j, cfg["fix"]["ai"], cfg["fix"]["u"]])
                elif cfg["x"] == "GDP" and cfg["y"] == "AI":
                    s = np.array([i, cfg["fix"]["d"], j, cfg["fix"]["u"]])
                else:
                    s = np.array([cfg["fix"]["g"], cfg["fix"]["d"], i, j])
                a, _ = get_policy(Q, s)
                pg[i, j] = a
        im = ax.imshow(pg, cmap=action_cmap, vmin=0, vmax=4, aspect="auto")
        ax.set_title(f"π*(s) — {cfg['title']}", fontweight="bold")
        ax.set_xlabel(cfg["x"]); ax.set_ylabel(cfg["y"])
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(LEVEL_L); ax.set_yticklabels(LEVEL_L)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, labels_short[pg[i,j]],
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="black")

    patches = [mpatches.Patch(color=action_cmap(i),
                              label=f"a{i}: {labels_short[i]}")
               for i in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def chart_policy_comparison(Q, results_dict):
    """Câu 11.3.4 — Violin + Cumulative reward rollout."""
    env    = VietnamEconomyEnv()
    policy_names = list(results_dict.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Violin plot
    ax = axes[0]
    data_list = [results_dict[k] for k in policy_names]
    parts = ax.violinplot(data_list, positions=range(len(policy_names)),
                          showmeans=True, showmedians=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_alpha(0.6)
    ax.set_xticks(range(len(policy_names)))
    ax.set_xticklabels(policy_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Tổng Reward / Episode")
    ax.set_title("Phân bố Reward — 4 chính sách (500 episodes)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for i, k in enumerate(policy_names):
        ax.text(i, np.mean(results_dict[k]),
                f"{np.mean(results_dict[k]):.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Cumulative reward rollout
    ax = axes[1]
    np.random.seed(99)
    policies_fn = {
        "π* (Q-learning)":     lambda s: int(np.argmax(Q[tuple(s)])),
        "Rule: a1 (Cân bằng)": lambda s: 1,
        "Rule: a3 (AI-dẫn)":   lambda s: 3,
        "Random":              lambda s: env.sample_action(),
    }
    for (name, fn), color in zip(policies_fn.items(), colors):
        np.random.seed(99)
        s = env.reset()
        cumulative, total_r = [], 0.0
        for t in range(env.T):
            a = fn(s)
            s, r, done = env.step(a)
            total_r += r
            cumulative.append(total_r)
        ax.plot(range(1, env.T+1), cumulative, "o-", color=color,
                lw=2, ms=6, label=f"{name} ({total_r:.3f})")
    ax.set_xlabel("Năm (t)")
    ax.set_ylabel("Reward tích lũy")
    ax.set_title("Reward tích lũy theo năm (1 episode mẫu)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, env.T+1))

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────
# 5. DQN (Stable-Baselines3) — tùy chọn
# ──────────────────────────────────────────────────────────────────
def try_run_dqn(Q, results_ql):
    """Câu 11.3.5 — DQN nếu SB3 có sẵn."""
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from gymnasium.spaces import Box
        from stable_baselines3 import DQN

        class VNEconFlat(gym.Env):
            metadata = {"render_modes": []}
            def __init__(self):
                super().__init__()
                self._base = VietnamEconomyEnv()
                self.action_space = spaces.Discrete(5)
                self.observation_space = Box(
                    low=np.zeros(4, dtype=np.float32),
                    high=np.ones(4, dtype=np.float32)*2, dtype=np.float32)
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                s = self._base.reset()
                return s.astype(np.float32), {}
            def step(self, action):
                s, r, done = self._base.step(action)
                return s.astype(np.float32), r, done, False, {}

        env_flat  = VNEconFlat()
        dqn_model = DQN(
            "MlpPolicy", env_flat,
            learning_rate=1e-3, batch_size=32, buffer_size=10000,
            exploration_initial_eps=1.0, exploration_final_eps=0.05,
            exploration_fraction=0.5, gamma=0.95,
            policy_kwargs={"net_arch":[64,64]},
            verbose=0, seed=42,
        )
        dqn_model.learn(total_timesteps=100_000, progress_bar=False)

        env_eval = VietnamEconomyEnv()
        dqn_rewards = []
        for _ in range(500):
            s = env_eval.reset()
            total_r = 0.0
            while True:
                a, _ = dqn_model.predict(s.astype(np.float32), deterministic=True)
                s, r, done = env_eval.step(int(a))
                total_r += r
                if done:
                    break
            dqn_rewards.append(total_r)
        return dqn_rewards, None
    except ImportError as e:
        return None, str(e)


# ──────────────────────────────────────────────────────────────────
# 6. HÀM RUN() CHÍNH
# ──────────────────────────────────────────────────────────────────
def run():
    st.title("📘 Bài 11 — Học Tăng Cường (Q-learning) "
             "cho Chính sách Kinh tế Thích nghi")
    st.warning(
        "⚠️ **Lưu ý:** AI hỗ trợ ra quyết định — không thay thế trách nhiệm "
        "chính trị-xã hội. Bài tập này nhằm minh họa kỹ thuật, không nhằm tự "
        "động hóa hoạch định chính sách trên thực tế."
    )

    st.markdown("""
    #### 🎯 Mục tiêu
    1. Mô hình hóa nền kinh tế như **Markov Decision Process (MDP)**
    2. Cài đặt **môi trường** mô phỏng kinh tế Việt Nam
    3. Huấn luyện **Q-learning tabular** (up to 10.000 episodes)
    4. Trích xuất **chính sách tối ưu π*(s)** và phân tích theo trạng thái
    5. **So sánh** Q-learning với 3 chính sách rule-based
    6. *(Mở rộng)* Thử **DQN** qua stable-baselines3

    #### 📐 Mô hình MDP

    | Thành phần | Chi tiết |
    |---|---|
    | **State** $s_t$ | (GDP growth, Digital, AI capacity, Unemp risk) × {low, mid, high} → $3^4=81$ trạng thái |
    | **Action** $a_t$ | 5 chiến lược phân bổ ngân sách |
    | **Reward** $R_t$ | $0.40\\Delta GDP - 0.25\\Delta U - 0.20\\cdot CyberRisk - 0.15\\cdot Emission$ |
    | **Discount** $\\gamma$ | 0.95 |
    | **Episode** | T = 10 năm |
    """)
    st.markdown("---")

    # ── Bước 1: Môi trường ──────────────────────────────────────────
    st.subheader("🌏 Câu 11.3.1 — Môi trường VietnamEconomyEnv")

    col1, col2, col3 = st.columns(3)
    col1.metric("Số trạng thái", "81 (3⁴)")
    col2.metric("Số hành động", "5 chiến lược")
    col3.metric("Độ dài episode", "T = 10 năm")

    df_actions = pd.DataFrame({
        "Action": ["a0","a1","a2","a3","a4"],
        "Chiến lược": ["Truyền thống","Cân bằng","Số hóa nhanh",
                       "AI dẫn dắt","Bao trùm"],
        "K%": [70,40,25,20,30],
        "D%": [10,25,45,20,20],
        "AI%":[10,15,15,45,10],
        "H%": [10,20,15,15,40],
    }).set_index("Action")
    st.dataframe(df_actions, use_container_width=True)

    # Chạy thử 1 episode random
    env_test = VietnamEconomyEnv()
    s_init = env_test.reset()
    st.info(f"Trạng thái khởi đầu VN 2026: **{env_test.state_desc(s_init)}**  \n"
            f"(GDP=medium, Digital=medium, AI=low, Unemp=medium)")

    st.markdown("---")

    # ── Bước 2: Tham số huấn luyện ──────────────────────────────────
    st.subheader("⚙️ Tham số Q-learning")
    col1, col2, col3, col4 = st.columns(4)
    N_EPISODES = col1.selectbox("Số episodes", [2000, 5000, 10000], index=2)
    ALPHA      = col2.slider("Learning rate α", 0.01, 0.5, 0.10, step=0.01)
    GAMMA      = col3.slider("Discount γ",      0.80, 0.99, 0.95, step=0.01)
    EPS_END    = col4.slider("ε cuối",          0.01, 0.2, 0.05, step=0.01)

    # ── Nút chạy ────────────────────────────────────────────────────
    if st.button("🚀 Huấn luyện Q-learning", type="primary"):

        # === CÂU 11.3.2 — Huấn luyện ===
        progress_bar = st.progress(0, text="Đang huấn luyện Q-learning…")
        Q, episode_rewards, episode_eps = run_model(
            n_episodes=N_EPISODES, alpha=ALPHA, gamma=GAMMA,
            eps_start=1.0, eps_end=EPS_END,
        )
        progress_bar.progress(100, text="Huấn luyện hoàn thành!")

        st.subheader("✅ Câu 11.3.2 — Kết quả huấn luyện Q-learning")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Episodes", f"{N_EPISODES:,}")
        col2.metric("Q-table shape", "3×3×3×3×5 = 405 entries")
        col3.metric("Non-zero Q entries", f"{(Q != 0).sum():,}")
        col4.metric("Avg reward (10% cuối)",
                    f"{np.mean(episode_rewards[-int(N_EPISODES*0.1):]):.4f}")

        # Learning curve
        fig_lc = chart_learning_curve(episode_rewards, episode_eps, N_EPISODES)
        st.pyplot(fig_lc)
        plt.close(fig_lc)

        st.markdown("---")

        # === CÂU 11.3.3 — Chính sách π*(s) ===
        st.subheader("🗺️ Câu 11.3.3 — Chính sách Tối ưu π*(s)")

        ACTION_NAMES = VietnamEconomyEnv.ACTION_NAMES
        LEVEL_LABELS = VietnamEconomyEnv.LEVEL_LABELS

        test_states = [
            {"label":"VN 2026 (thực tế)",  "state":[1,1,0,1],
             "desc":"GDP=mid, D=mid, AI=low, U=mid"},
            {"label":"Khủng hoảng",         "state":[0,0,0,2],
             "desc":"GDP=low, D=low, AI=low, U=high"},
            {"label":"Tăng trưởng mạnh",    "state":[2,1,1,0],
             "desc":"GDP=high, D=mid, AI=mid, U=low"},
            {"label":"Đã số hóa cao",       "state":[1,2,2,0],
             "desc":"GDP=mid, D=high, AI=high, U=low"},
            {"label":"Rủi ro thất nghiệp",  "state":[1,0,1,2],
             "desc":"GDP=mid, D=low, AI=mid, U=high"},
        ]

        policy_rows = []
        for ts in test_states:
            s = np.array(ts["state"])
            best_a, q_vals = get_policy(Q, s)
            policy_rows.append({
                "Kịch bản": ts["label"],
                "Trạng thái": ts["desc"],
                "Hành động tối ưu": ACTION_NAMES[best_a],
                "Q-values (a0..a4)": [round(v,3) for v in q_vals],
            })
        df_policy = pd.DataFrame(policy_rows).set_index("Kịch bản")
        st.dataframe(df_policy[["Trạng thái","Hành động tối ưu"]],
                     use_container_width=True)

        # Chi tiết Q-values
        with st.expander("🔍 Chi tiết Q-values từng kịch bản"):
            for row in policy_rows:
                st.write(f"**{row['Kịch bản']}** ({row['Trạng thái']})")
                q_df = pd.DataFrame({
                    "Action": ACTION_NAMES,
                    "Q-value": row["Q-values (a0..a4)"],
                })
                q_df["Tối ưu?"] = ["✅" if i == q_df["Q-value"].idxmax() else ""
                                    for i in range(len(ACTION_NAMES))]
                st.dataframe(q_df.set_index("Action"), use_container_width=True)

        # Phân bố hành động qua 81 trạng thái
        all_best = []
        for g in range(3):
            for d in range(3):
                for ai in range(3):
                    for u in range(3):
                        a, _ = get_policy(Q, np.array([g,d,ai,u]))
                        all_best.append(a)
        action_counts = np.bincount(all_best, minlength=5)

        st.markdown("**Phân bố hành động tối ưu qua 81 trạng thái:**")
        df_dist = pd.DataFrame({
            "Hành động": ACTION_NAMES,
            "Số trạng thái": action_counts,
            "Tỷ lệ (%)": [round(c/81*100, 1) for c in action_counts],
        }).set_index("Hành động")
        st.dataframe(df_dist, use_container_width=True)

        # Heatmap
        fig_hm = chart_policy_heatmap(Q)
        st.pyplot(fig_hm)
        plt.close(fig_hm)

        st.markdown("---")

        # === CÂU 11.3.4 — So sánh chính sách ===
        st.subheader("⚔️ Câu 11.3.4 — So sánh π* với 3 Chính sách Rule-Based")

        env_cmp = VietnamEconomyEnv()
        policies = {
            "π* (Q-learning)":     lambda s: int(np.argmax(Q[tuple(s)])),
            "Rule: a1 (Cân bằng)": lambda s: 1,
            "Rule: a3 (AI-dẫn)":   lambda s: 3,
            "Random":              lambda s: env_cmp.sample_action(),
        }

        with st.spinner("Đang đánh giá 4 chính sách (500 episodes mỗi loại)…"):
            results = {name: evaluate_policy(fn, n_eval=500, seed=0)
                       for name, fn in policies.items()}

        comp_rows = []
        for name, rlist in results.items():
            comp_rows.append({
                "Chính sách": name,
                "Mean Reward": round(np.mean(rlist), 4),
                "Std":         round(np.std(rlist), 4),
                "Min":         round(np.min(rlist), 4),
                "Max":         round(np.max(rlist), 4),
            })
        df_comp = pd.DataFrame(comp_rows).set_index("Chính sách")
        st.dataframe(df_comp, use_container_width=True)

        ql_mean   = np.mean(results["π* (Q-learning)"])
        best_rule = max(["Rule: a1 (Cân bằng)", "Rule: a3 (AI-dẫn)"],
                        key=lambda k: np.mean(results[k]))
        rule_mean = np.mean(results[best_rule])
        improvement = (ql_mean - rule_mean) / abs(rule_mean) * 100
        st.metric(f"Q-learning vs best rule ({best_rule})",
                  f"{ql_mean:.4f}", f"{improvement:+.2f}%")

        fig_cmp = chart_policy_comparison(Q, results)
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        st.markdown("---")

        # === CÂU 11.3.5 — DQN (tùy chọn) ===
        st.subheader("🧠 Câu 11.3.5 (Mở rộng) — Deep Q-Network (DQN)")
        st.markdown(
            "DQN thay thế Q-table bằng **neural network** "
            "(2 lớp ẩn × 64 units) → xử lý tốt không gian trạng thái lớn."
        )

        if st.checkbox("Chạy DQN (cần stable-baselines3 + gymnasium, ~100K timesteps)"):
            with st.spinner("Đang huấn luyện DQN (100,000 timesteps)…"):
                dqn_rewards, dqn_err = try_run_dqn(Q, results)

            if dqn_err:
                st.error(
                    f"Không thể chạy DQN: {dqn_err}  \n"
                    "Cài bằng: `pip install stable-baselines3 gymnasium`"
                )
            else:
                col1, col2 = st.columns(2)
                col1.metric("DQN mean reward",
                            f"{np.mean(dqn_rewards):.4f}",
                            f"{np.mean(dqn_rewards)-ql_mean:+.4f} vs Q-learning")
                col2.metric("Q-learning mean reward", f"{ql_mean:.4f}")

                dqn_diff = np.mean(dqn_rewards) - ql_mean
                if abs(dqn_diff) < 0.01:
                    st.info(
                        "💡 Với 81 trạng thái, Q-learning tabular đủ tốt. "
                        "DQN có lợi thế khi không gian trạng thái lớn (continuous state)."
                    )
                elif dqn_diff > 0:
                    st.success(
                        f"✅ DQN cải thiện thêm {dqn_diff:+.4f} "
                        f"({dqn_diff/abs(ql_mean)*100:+.2f}%) so với Q-learning."
                    )
                else:
                    st.warning(
                        f"DQN chưa vượt Q-learning ({dqn_diff:+.4f}). "
                        "Thử tăng timesteps hoặc tuning hyperparameters."
                    )

        st.markdown("---")

        # === TỔNG KẾT ===
        st.subheader("📋 Tổng kết Bài 11")

        best_a_vn, q_vals_vn = get_policy(Q, np.array([1,1,0,1]))

        st.markdown(f"""
        | Chỉ tiêu | Kết quả |
        |---|---|
        | **Số episodes** | {N_EPISODES:,} |
        | **Avg reward (10% cuối)** | {np.mean(episode_rewards[-int(N_EPISODES*0.1):]):.4f} |
        | **Non-zero Q entries** | {(Q!=0).sum():,} / 405 |
        | **π*(VN 2026)** | {ACTION_NAMES[best_a_vn]} |
        | **π*(Khủng hoảng)** | {ACTION_NAMES[get_policy(Q, np.array([0,0,0,2]))[0]]} |
        | **π*(Tăng trưởng mạnh)** | {ACTION_NAMES[get_policy(Q, np.array([2,1,1,0]))[0]]} |
        | **Q-learning vs best rule** | {improvement:+.2f}% |
        """)

        with st.expander("💬 Câu hỏi thảo luận"):
            a_crisis, _ = get_policy(Q, np.array([0,0,0,2]))
            a_growth, _ = get_policy(Q, np.array([2,1,1,0]))
            st.markdown(f"""
            **a)** Khi khủng hoảng (GDP=low, D=low, U=high):
            π* chọn **{ACTION_NAMES[a_crisis]}**.
            *Gợi ý phân tích: Quick win có nghĩa chọn hành động có kết quả nhanh
            (giảm thất nghiệp, ổn định). Bao trùm (a4) ưu tiên nhân lực giúp giảm U.*

            **b)** Khi tăng trưởng mạnh (GDP=high, AI=mid, U=low):
            π* chọn **{ACTION_NAMES[a_growth]}**.
            *Gợi ý: Consolidation = đầu tư dài hạn. Khi ổn định, đầu tư AI hoặc số hóa
            để duy trì đà tăng trưởng.*

            **c)** Tích hợp π* vào hoạch định chính sách:
            π* → Báo cáo tư vấn → Hội đồng chính sách → Phê duyệt → Thực thi.
            AI đóng vai trò phân tích và gợi ý — **không quyết định thay con người.**
            """)
if __name__ == "__main__":
    run()