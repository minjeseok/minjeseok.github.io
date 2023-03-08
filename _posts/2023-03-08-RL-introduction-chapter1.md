---
layout: article
title: Chapter 1. The Reinforcement Learning Problem
aside:
  toc: true
sidebar:
  nav: layouts
---

# 1. The Reinforcement Learning Problem
## 1.1 Reinforcement Learning
머신러닝은 지도학습(Supervised), 비지도학습(Unsupervised), 강화학습(Reinforced) 크게 세 가지로 분류된다. 이들의 궁극적인 목적으로는 지도학습은 결과를 예측(Regression, Classification)하는 것이고, 비지도 학습은 데이터의 본질적 패턴을 발견(Association, Clustering)하는 것이다. 이와 다르게 강화학습은 환경과의 상호작용을 통해 얻은 보상을 통해 일련의 행동을 학습하는 것을 목적으로 한다. 행동에 대한 결과와 보상은 궁극적인 목표를 달성하기 위한 풍부한 정보를 내포하며, 환경이 자신의 행동에 대해 어떻게 변화하는지 학습할 수 있다. 상호작용이란 강화학습에서 모든 학습 이론의 기반이 되는 기본 아이디어라고 볼 수 있다. 

 

### 1.1.1 Characteristics of Reinforcement Learning Problem 
__- Closed-loop이다.__

Closed-loop란 사람의 개입없이 원하는 상태 유지를 위해 시스템을 자동으로 조절하는 방식을 의미한다. 강화학습은 본질적으로 사람과의 상호작용이 없을 뿐더러 학습자의 동작 이후의 결과만이 입력에 영향을 미치게 된다. 

 

__- 학습자에게 직접적인 지침이 주어지지 않는다.__

학습자는 상호작용 속에서 수치적 보상 신호를 최대화 하기 위해 현재의 환경에서 어떤 행동을 수행해야하는지 배워야 한다. 이 때, 학습자는 trial & error 방식으로 즉, 여러 시행착오를 통해 어떤 행동이 가장 큰 보상을 얻을 수 있는지 발견한다.

 

__- 보상 신호를 포함한 행동의 결과가 장기간에 걸쳐 나타난다.__

행동에 대한 보상은 즉각적인 보상뿐만 아니라 다음 상황과 그로 발생하는 모든 후속 보상에 영향을 미칠 수 있다. 추후 고려하겠지만, 이러한 특성으로 인해 학습자는 다음 1-step 뒤의 즉각적인 보상만을 목적하기도, 혹은 n-step 뒤의 장기적인 보상을 목적할 수도 있다. 

 

### 1.1.2 Exploration vs. Exploitation
강화학습에서 발생하는 문제는 exploration(탐색)과 exploitation(착취) 간의 균형이다. 학습자가 과거에 시도했고 보상을 생성하는데 효과적인 것으로 확인된 행동을 선호해야 하는데, 이러한 행동 발견을 위해서는 이전에 선택하지 않았던 행동을 시도해야하는 딜레마에 빠지게 된다. 즉, 학습자는 이전에 알고있던 보상 정보를 exploit해야 하지만, 더 나은 보상의 행동 선택을 하기 위해 explore 해야한다. 

 

### 1.1.3 Characteristics of Reinforcement Learning
강화학습은 전체가 아닌 부분적으로만 관찰된다던가 하는 불확실한 환경과 상호작용하는 목표 지향 학습자의 전체 문제를 명시적으로 고려한다. 모든 강화학습 학습자는 명확한 목표를 가지고 있고, 또한 환경이 불확실하더라도 명확한 목표를 위해 동작한다고 가정된다. 가령, 어떤 하위 문제로부터 결과가 도출되었는지 확인하기 위한 연구에서는 완전하게 상호 작용하며 목표를 추구하는 에이전트 내에서 명확한 역할을 수행하는 하위 문제가 필요하다. 다시 말해, 유용한 결과를 산출한 전체 문제라도 그 하위 문제에 초점을 맞추려면 이를 분리시켜야 하는데, 강화학습은 하위 문제에서도 명시적으로 특정 목표를 추구하기 때문에 하위 문제가 전체 문제에 어떻게 맞춰지는지 조사할 수 있는 근거가 된다. 

 

### 1.1.4 Trends in Reinforcement Learning
강화학습은 엔지니어링 및 과학 분야에서 유익한 상호작용을 거치며 발전해 나가고 있다. 통계, 최적화와 같은 수학적 주제를 넘어 매개변수화된 근사기의 적용과 같은 융합은 여러 패러다임을 열며 모델들의 성능을 향상시켰다. 그러나 과거로부터의 이러한 발전은 각 도메인에 치중하여 많은 특수 절차 및 휴리스틱을 가정했고 이는 일반적인 원칙을 찾지 않아왔다. 최신 AI는 이제 학습과 의사결정의 일반 원칙을 찾고 도메인 지식을 통합하기 위한 노력을 하고 있다. 


https://drive.google.com/file/d/1l906U4WOc_AossEWdSbsj52Rc8X0QCYX/view?usp=share_link
<img class="image image--lg" src="[https://drive.google.com/file/d/1l906U4WOc_AossEWdSbsj52Rc8X0QCYX/view?usp=share_link](https://drive.google.com/uc?export=view&id=1l906U4WOc_AossEWdSbsj52Rc8X0QCYX"/>

[gamer_example](https://drive.google.com/uc?export=view&id=1l906U4WOc_AossEWdSbsj52Rc8X0QCYX)



