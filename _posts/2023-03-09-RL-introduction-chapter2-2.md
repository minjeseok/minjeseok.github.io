"강화학습의 바이블이라고 불리는 Richard S. Sutton의 Reinforcement Learning: An Introduction을 공부하고 필자가 이해한 내용과 추가 정보들을 예제와 함께 간략하고 쉽게(?) 설명하는 방식으로 정리해봅니다. 용어 같은 경우, 원문 그대로 사용하겠지만 혹시 모를 번역 오류 및 잘못된 설명에 대한 지적과 보충 환영합니다."


# 2. Multi-arm Bandits (2)



## 2.5 Optimistic Initial Values
이전 게시물에서 우리는 initial action-value estimates $Q_1(a)$ 값에 의존해왔다. 쉽게 말하면, 우리는 맨 처음 설정된 초기 action-value function에 의해 동작하게 되며 이는 곧 편향됨을 의미한다. sample average 방법의 경우에는 모든 action이 한번 이상 선택되면 편향이 사라지지만, constant $\alpha$를 사용하는 경우 편향은 사라지지않지만 시간이 지남에 따라 감소하게 된다. initial action-value의 편향은 예상되는 reward 수준에 대한 사전 지식을 제공하기도 하지만, 이 또한 사용자가 설정하는 파라미터로 취급될 수도 있다는 단점이 존재한다. 



### 2.5.1 Using Initial Action Value to Exploration 
편향성을 지닌 initial action-value는 exploration은 장려하는 방식으로 사용될 수 있다. 10-armed testbed에서 initial action-value를 0으로 설정하는 대신 +5로 설정했다고 가정한다. 이 문제의 $q(a)$는 $\mu=0, \sigma=1$인 starnard distribution이었는데, 따라 initial estimate는 optimistic하다고 여겨지고 action-value 방법을 explore하도록 권장한다. 어떤 action을 선택해도 reward는 initial action value보다 적기 때문에 agent는 더 나은 action을 선택하려고 한다. 결과적으로 value estimate가 수렴하기 전에 모든 action이 여러번 시도되며 greedy action이 매번 선택되더라도 agent는 상당한 exploration을 수행하게 된다. 



아래 그래프는 모든 $a$에 대해 $Q_1(a) = +5$를 사용하는 greedy 방법과 $Q_1(a) = 0$인 $\epsilon$-greedy를 비교한 10-armed bandit testbed의 성능을 보여준다. 

<center><img src="https://user-images.githubusercontent.com/127313067/223759254-514685c8-89f3-425e-b514-18376c781289.png" width="60%" height="60%"></center>



초기 optimistic 방법이 더 많이 explore 하기 때문에 성능이 좋지 않지만 시간이 지남에 따라 explore이 줄기 때문에 $\epsilon$-greedy보다도 성능이 더 좋아진다. 한눈에 보기에는 매우 좋은 방법처럼 보일 수 있으나 이는 stationary 문제에만 적용이 가능하다. 즉, action-value가 변경되는 non-stationary env에서는 적합하지 않다. 이러한 관점은 모든 후속 rewards를 동일하게 평균화하는 sample average 방법에도 동일하게 적용된다. 그럼에도 불구하고 이를 기반으로 한 방법들은 매우 단순하면서도 종종 적합한 경우도 있다. 



## 2.6 Upper-Confidence-Bound Action Selection
action value estimate가 부정확함으로 인해 exploration이 필요하다. $\epsilon$-greedy를 수행할수도 있지만 이는 불확실한 action에 대한 preference 없이 무차별적으로 시도된다. 우리는 estimate가 maximal과 얼마나 가까운지와 estimate의 불확실성을 모두 고려하여 실제로 optimal일 가능성에 따라 non-greedy action 중에서 선택하는 것이 바람직하다. 



이를 효과적으로 수행하기 위해서는 action을 다음과 같이 선택한다. 여기서 $\ln t$는 $t$($e \approx 2.71828$)의 natural logarithm를 나타내며 $c > 0 $는 exploration 정도를 제어한다. 만약 $N_t(a) =0$이면, $a$는 maximizing action이라고 간주된다. 



$$ A_t = \arg\max_a \left[ {Q_t(a) + c\sqrt{\cfrac{\ln t}{N_t(a)}}} \, \right ] \label{8}\tag{8}$$

upper confidence bound(UCB)의 아이디어는 불확실성 또는 variance의 척도를 의미하는 squae-root 항을 사용하여 $a$ 값 estimate를 표현하자는 것이다. 따라서 해당 action value가 최대가 되는 값은 신뢰 수준을 결정하는 $c$와 함께 action $a$의 가능한 true value에 대한 upper bound이다. $a$가 선택될 때마다 불확실성은 감소할 것이다. action $a$가 선택된 횟수인 $N_t(a)$는 증가하고 이는 불확실성 항의 분모에 나타나므로 항은 감소한다. 반면에 다른 $a$가 선택될 때마다 $t$는 증가한다. 분자에 나타나므로 불확실성 추정치가 증가한다. natural logrithm의 사용은 증가폭이 시간이 지남에 따라 작아지지만 제한이 없음을 의미한다. 결국 모든 action이 선택되지만 시간이 지남에 따라 estimate가 낮거나 이미 더 많이 선택된 action의 경우 대기 시간이 길어지고 선택 빈도가 낮아지게 된다.


<center><img src="https://user-images.githubusercontent.com/127313067/223759350-6748d566-73dd-4220-8f29-321cbecfb6b8.png" width="60%" height="60%"></center>



10-armed testbed에서 UCB를 사용한 결과이다. UCB는 종종 잘 수행되지만 bandit 문제 이외에서는 강화학습에서 일반적인 설정이 아닌 stationary env 성질로 인해 다른 문제들로의 확장은 어렵다. 또한 나중에 배우게 될 large state space, 특히 function approximation에서의 적용도 어렵다. 이러한 고급 설정에서는 UCB 아이디어를 활용하는 실용적인 방법은 존재하지 않는다고 한다. 



## 2.7 Gradient Bandits
여태까지는 action-value를 추정하고 해당 estimate를 사용하여 action을 선택하는 방법을 고려했으나, 이 섹션에서는 각 action $a$에 대한 numerical preference $H_t(a)$ 학습을 고려해본다. preference가 클수록 해당 action이 더 자주 수행되지만 preference는 reward 측면에서 해석되지 않고 오직 한 action이 다른 action보다 상대적으로 preference 되는 것 만이 중요하다. 모든 preference에 1000을 추가하여 다음과 같이 softmax distribution(i.e. Gibbs or Boltzmann distribution)에 따라 결정되는 action probability에 영향을 미치지 않게 하였고 이는 아래와 같다. 



$$ \Pr \{ {A_t = a} \} = \cfrac{e^{H_t(a)}}{\sum^n_{b=1}e^{H_t(b)}} = \pi_t(a) $$

여기서 $\pi_t(a)$는 time $t$에 action $a$를 수행할 확률을 의미한다. 초기 모든 preferences는 같다(e.g. $H_1(a) = 0, \forall a$). 



$$ \begin{align*} H_{t+1}(A_t) &= H_t(A_t) + \alpha(R_t-\barR_t)(1-\pi_t(A)t)), \, and\\
H_{t+1}(a) &= H_t(a) - \alpha(R_t-\bar R_t)\pi_t(a), \quad\quad\quad \forall a \ne A_t \label\end{align*} $$
