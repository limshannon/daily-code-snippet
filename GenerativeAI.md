All Readings: Introduction to Generative AI (G-GENAI-I)

Here are the assembled readings on generative AI:
● Ask a Techspert: What is generative AI?
https://blog.google/inside-google/googlers/ask-a-techspert/what-is-generative-ai/
● Build new generative AI powered search & conversational experiences with Gen App

Builder:
https://cloud.google.com/blog/products/ai-machine-learning/create-generative-apps-inminutes-with-gen-app-builder
● What is generative AI?
https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-generative-ai
● Google Research, 2022 & beyond: Generative models:
https://ai.googleblog.com/2023/01/google-research-2022-beyond-language.html#GenerativeModels
● Building the most open and innovative AI ecosystem:
https://cloud.google.com/blog/products/ai-machine-learning/building-an-open-generative-ai-partner-ecosystem
● Generative AI is here. Who Should Control It?
https://www.nytimes.com/2022/10/21/podcasts/hard-fork-generative-artificial-intelligence.html
● Stanford U & Google’s Generative Agents Produce Believable Proxies of Human
Behaviors:
https://syncedreview.com/2023/04/12/stanford-u-googles-generative-agents-produce-believable-proxies-of-human-behaviours/
● Generative AI: Perspectives from Stanford HAI:
https://hai.stanford.edu/sites/default/files/2023-03/Generative_AI_HAI_Perspectives
● Generative AI at Work:
https://www.nber.org/system/files/working_papers/w31161/w31161.pdf
● The future of generative AI is niche, not generalized:
https://www.technologyreview.com/2023/04/27/1072102/the-future-of-generative-ai-isniche-not-generalized/
● The implications of Generative AI for businesses:
https://www2.deloitte.com/us/en/pages/consulting/articles/generative-artificial-intelligence.html
● Proactive Risk Management in Generative AI:
https://www2.deloitte.com/us/en/pages/consulting/articles/responsible-use-of-generati
ve-ai.html
● How Generative AI Is Changing Creative Work:
https://hbr.org/2022/11/how-generative-ai-is-changing-creative-work
Here are the assembled readings on large language models:
● NLP's ImageNet moment has arrived: https://thegradient.pub/nlp-imagenet/
● LaMDA: our breakthrough conversation technology:
https://blog.google/technology/ai/lamda/
● Language Models are Few-Shot Learners:
https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64aPaper.pdf
● PaLM-E: An embodied multimodal language model:
https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html
● PaLM API & MakerSuite: an approachable way to start prototyping and building
generative AI applications:
https://developers.googleblog.com/2023/03/announcing-palm-api-and-makersuite.html
● The Power of Scale for Parameter-Efficient Prompt Tuning:
https://arxiv.org/pdf/2104.08691.pdf
● Google Research, 2022 & beyond: Language models:
https://ai.googleblog.com/2023/01/google-research-2022-beyond-language.html/Langu
ageModels
● Solving a machine-learning mystery:
https://news.mit.edu/2023/large-language-models-in-context-learning-0207
Additional Resources:
● Attention is All You Need: https://research.google/pubs/pub46201/
● Transformer: A Novel Neural Network Architecture for Language Understanding:
https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html
● Transformer on Wikipedia:
https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#:~:text=Transformers%20were%20introduced%20in%202017,allowing%20training%20on%20larger%20datasets.
● What is Temperature in NLP? https://lukesalamone.github.io/posts/what-is-temperature/
● Model Garden: https://cloud.google.com/model-garden
● Auto-generated Summaries in Google Docs:
https://ai.googleblog.com/2022/03/auto-generated-summaries-in-google-docs.html



Generative AI 소개에 오신 것을 환영합니다.
이 과정에서는 생성적 AI를 정의하고, 생성적 AI의 작동 방식을 설명하고, 생성적 AI 모델 유형을 설명하고, 생성적 AI 애플리케이션을 설명하는 방법을 배웁니다.
제너레이티브 AI(Generative AI)는 텍스트, 이미지, 오디오, 합성 데이터 등 다양한 형태의 콘텐츠를 생산할 수 있는 인공지능 기술의 일종이다.

그런데 인공지능이란 무엇일까?
글쎄요, 우리는 생성 인공 지능을 탐구할 것이므로 약간의 맥락을 제공하겠습니다.
그래서 가장 많이 묻는 두 가지 질문은 인공 지능이 무엇인지, AI와 기계 학습의 차이점이 무엇인지입니다.
그것에 대해 생각하는 한 가지 방법은 AI가 예를 들어 물리학과 같은 학문이라는 것입니다.
AI는 자율적으로 추론하고, 학습하고, 행동할 수 있는 시스템인 정보 에이전트의 생성을 다루는 컴퓨터 과학의 한 분야입니다.
본질적으로 AI는 인간처럼 생각하고 행동하는 기계를 만들기 위한 이론 및 방법과 관련이 있습니다.
이 분야에는 AI의 하위 분야인 머신러닝이 있습니다.

입력 데이터로부터 모델을 훈련시키는 프로그램이나 시스템입니다.
훈련된 모델은 모델을 훈련하는 데 사용된 것과 동일한 데이터에서 가져온 새로운 데이터 또는 이전에 본 적이 없는 데이터로부터 유용한 예측을 할 수 있습니다.
기계 학습은 컴퓨터에 명시적인 프로그래밍 없이도 학습할 수 있는 기능을 제공합니다.
기계 학습 모델의 가장 일반적인 두 가지 클래스는 비지도 ML 모델과 지도 ML 모델입니다.
둘 사이의 주요 차이점은 감독 모델의 경우 레이블이 있다는 것입니다.
레이블이 지정된 데이터는 이름, 유형 또는 숫자와 같은 태그와 함께 제공되는 데이터입니다.
레이블이 없는 데이터는 태그 없이 제공되는 데이터입니다.
이 그래프는 지도 모델이 해결하려고 시도할 수 있는 문제의 예입니다.

예를 들어 당신이 식당의 주인이라고 가정해보자.
청구 금액, 주문 유형, 픽업 또는 배송 여부에 따라 팁을 준 사람의 수에 대한 기록 데이터가 있습니다.
지도 학습에서 모델은 과거 사례를 통해 학습하여 미래 값(이 경우 팁)을 예측합니다.
따라서 여기서 모델은 총 청구 금액을 사용하여 주문이 픽업되었는지 또는 배달되었는지에 따라 향후 팁 금액을 예측합니다.
이는 비지도 모델이 해결하려고 시도할 수 있는 문제의 예입니다.
따라서 여기서 재직 기간과 수입을 살펴본 다음 직원을 그룹화하거나 클러스터링하여 누군가가 빠른 길에 있는지 확인하려고 합니다.
비지도 문제는 원시 데이터를 보고 그것이 자연스럽게 그룹에 속하는지 확인하는 발견에 관한 것입니다.
이러한 개념을 이해하는 것이 생성 AI를 이해하는 기초이므로 조금 더 깊이 들어가 이를 그래픽으로 보여드리겠습니다.

지도 학습에서는 테스트 데이터 값 또는 x가 모델에 입력됩니다.
모델은 예측을 출력하고 해당 예측을 모델 훈련에 사용된 훈련 데이터와 비교합니다.
예측된 테스트 데이터 값과 실제 학습 데이터 값이 멀리 떨어져 있는 경우 이를 오류라고 합니다.
그리고 모델은 예측값과 실제값이 더 가까워질 때까지 이 오류를 줄이려고 합니다.
이것은 전형적인 최적화 문제입니다.
이제 우리는 인공 지능과 기계 학습의 차이점을 살펴보고 감독 및
비지도 학습을 통해 머신러닝 방법의 하위 집합으로 딥러닝이 적합한 위치를 간략하게 살펴보겠습니다.
머신러닝이 다양한 기술을 포괄하는 광범위한 분야인 반면, 딥러닝은 일종의 기술입니다.
인공 신경망을 활용해 머신러닝보다 더 복잡한 패턴을 처리하는 머신러닝의 일종이다.
인공 신경망은 인간의 두뇌에서 영감을 받았습니다.
이는 데이터를 처리하고 예측을 통해 작업을 수행하는 방법을 배울 수 있는 상호 연결된 많은 노드 또는 뉴런으로 구성됩니다.
딥 러닝 모델에는 일반적으로 여러 계층의 뉴런이 있으므로 기존 기계 학습 모델보다 더 복잡한 패턴을 학습할 수 있습니다.
그리고 신경망은 레이블이 있는 데이터와 레이블이 없는 데이터를 모두 사용할 수 있습니다.
이를 준지도 학습이라고 합니다.
준지도 학습에서는 신경망이 소량의 레이블이 지정된 데이터와 대량의 레이블이 없는 데이터에 대해 훈련됩니다.
레이블이 지정된 데이터는 신경망이 작업의 기본 개념을 학습하는 데 도움이 되는 반면, 레이블이 지정되지 않은 데이터는 신경망이 새로운 사례로 일반화하는 데 도움이 됩니다.
이제 우리는 마침내 생성 AI가 이 AI 분야에 적합한 위치에 도달했습니다.
Gen AI는 딥 러닝의 하위 집합입니다. 즉, 인공 신경망을 사용하고 지도, 비지도, 준지도 방법을 사용하여 레이블이 지정된 데이터와 레이블이 지정되지 않은 데이터를 모두 처리할 수 있습니다.
대규모 언어 모델도 딥 러닝의 하위 집합입니다.
딥 러닝 모델, 즉 일반적으로 머신 러닝 모델은 생성 모델과 차별 모델의 두 가지 유형으로 나눌 수 있습니다.
판별 모델은 데이터 포인트의 레이블을 분류하거나 예측하는 데 사용되는 모델 유형입니다.
판별 모델은 일반적으로 레이블이 지정된 데이터 포인트의 데이터 세트에 대해 훈련됩니다.
그리고 데이터 포인트의 특징과 라벨 간의 관계를 배웁니다.
판별 모델이 훈련되면 이를 사용하여 새로운 데이터 포인트의 레이블을 예측할 수 있습니다.
생성 모델은 기존 데이터의 학습된 확률 분포를 기반으로 새로운 데이터 인스턴스를 생성합니다.
따라서 생성 모델은 새로운 콘텐츠를 생성합니다.
여기에서 이 예를 들어보세요.
판별 모델은 조건부 확률 분포 또는 주어진 x에 대한 출력인 y의 확률을 학습합니다.
우리의 입력에 따르면 이것이 개이고 고양이가 아닌 개로 분류됩니다.
생성 모델은 결합 확률 분포 또는 x와 y의 확률을 학습하고 예측합니다.
이것이 개라는 조건부 확률은 개의 그림을 생성할 수 있습니다.
요약하자면, 생성 모델은 새로운 데이터 인스턴스를 생성할 수 있는 반면, 판별 모델은 다양한 종류의 데이터 인스턴스를 구별합니다.
상단 이미지는 데이터와 레이블 간의 관계 또는 예측하려는 내용을 학습하려고 시도하는 기존 기계 학습 모델을 보여줍니다.
하단 이미지는 새로운 콘텐츠를 생성할 수 있도록 콘텐츠의 패턴을 학습하려고 시도하는 생성적 AI 모델을 보여줍니다.
이 그림에는 Gen AI와 그렇지 않은 것을 구별하는 좋은 방법이 나와 있습니다.
출력, y 또는 레이블이 숫자나 클래스(예: 스팸 여부 또는 확률)인 경우 AI 생성이 아닙니다.
예를 들어 출력이 음성이나 텍스트, 이미지 또는 오디오와 같은 자연어인 경우 이는 Gen AI입니다.
이를 수학적으로 시각화하면 다음과 같습니다.
한동안 이것을 보지 못했다면 y는 x의 f와 같습니다. 방정식은 다른 입력이 주어지면 프로세스의 종속 출력을 계산합니다.
y는 모델 출력을 나타냅니다.
f는 계산에 사용된 함수를 구현합니다.
그리고 x는 공식에 사용된 입력을 나타냅니다.
따라서 모델 출력은 모든 입력의 함수입니다.
y가 예상 판매량처럼 숫자라면 Gen AI가 아닙니다.
y가 판매 정의와 같은 문장인 경우 질문이 텍스트 응답을 유도하므로 생성적입니다.
응답은 모델이 이미 훈련한 모든 대규모 데이터를 기반으로 합니다.
높은 수준에서 요약하면 기존의 지도 학습 및 비지도 학습 프로세스에서는 훈련 코드와 레이블 데이터를 사용하여 모델을 구축합니다.
사용 사례나 문제에 따라 모델이 예측을 제공할 수 있습니다.
무언가를 분류하거나 클러스터링할 수 있습니다.
우리는 이 예를 사용하여 Gen AI 프로세스가 얼마나 더 강력한지 보여줍니다.
Gen AI 프로세스는 모든 데이터 유형의 훈련 코드, 레이블 데이터 및 레이블이 지정되지 않은 데이터를 가져와 기초 모델을 구축할 수 있습니다.
그러면 기초 모델이 새로운 콘텐츠를 생성할 수 있습니다.
예를 들어 텍스트, 코드, 이미지, 오디오, 비디오 등이 있습니다.
우리는 전통적인 프로그래밍에서 신경망, 생성 모델에 이르기까지 오랜 발전을 이루었습니다.
전통적인 프로그래밍에서는 항목을 구별하기 위한 규칙을 하드 코딩해야 했습니다.
고양이-- 유형, 동물; 다리는 4개; 귀, 두 개; 모피, 그렇죠; 털실과 캣닢을 좋아해요. 에서
신경망의 물결을 통해 우리는 고양이와 개의 사진을 네트워크에 제공하고 질문할 수 있습니다.
이것은 고양이이고 고양이를 예측할 것입니다. 생성 파동에서 우리는
사용자는 텍스트, 이미지, 오디오, 비디오 등 자체 콘텐츠를 생성할 수 있습니다.
PaLM, Pathways 언어 모델, LaMDA, 대화 애플리케이션용 언어 모델과 같은 예시 모델, 수집
인터넷의 여러 소스에서 얻은 매우 큰 데이터를 수집하고 기초 언어 모델을 구축합니다.
프롬프트에 입력하든 구두로 입력하든 간단히 질문하여 사용할 수 있습니다.
프롬프트 자체에 대해 이야기합니다. 그래서 당신이 고양이가 무엇인지 물어보면, 고양이는 이렇게 답할 수 있습니다.
고양이에 대해 배운 모든 것. 이제 우리는 공식적인 정의에 이르렀습니다. 무엇
생성 AI는? 젠AI(Gen AI)는 인공지능의 일종으로, 이를 기반으로 새로운 콘텐츠를 만들어내는 인공지능이다.
기존 콘텐츠에서 배운 내용에 대해 설명합니다. 기존 콘텐츠에서 학습하는 과정은 다음과 같습니다.
훈련이라고 하며 프롬프트가 제공되면 통계 모델이 생성됩니다. 일체 포함
모델을 사용하여 예상되는 응답이 무엇인지 예측하고 이를 통해 새로운 콘텐츠가 생성됩니다.
기본적으로 데이터의 기본 구조를 학습한 다음 다음과 같은 새로운 샘플을 생성할 수 있습니다.
훈련된 데이터와 유사합니다. 앞서 언급했듯이 생성 언어 모델은
보여진 예에서 배운 내용을 취하여 완전히 무언가를 만들 수 있습니다.
그 정보를 바탕으로 새로 만들어졌습니다. 대규모 언어 모델은 생성적 AI의 한 유형입니다.
자연스러운 언어 형태로 새로운 텍스트 조합을 생성합니다. 생성 이미지 모델
이미지를 입력으로 받아 텍스트, 다른 이미지 또는 비디오를 출력할 수 있습니다. 예를 들어,
출력 텍스트, 출력 이미지, 이미지 완성 아래에서 시각적 질문 답변을 얻을 수 있습니다.
생성됩니다. 그리고 출력 영상 아래에 애니메이션이 생성됩니다. 생성 언어 모델은 텍스트를 다음과 같이 사용합니다.
더 많은 텍스트, 이미지, 오디오 또는 결정을 입력하고 출력할 수 있습니다. 예를 들어, 출력 아래
텍스트, 질문 답변이 생성됩니다. 그리고 출력 이미지 아래에 비디오가 생성됩니다. 우리는 다음과 같이 말했습니다.

생성 언어 모델은 훈련 데이터를 통해 패턴과 언어에 대해 학습한 다음, 일부 텍스트가 주어지면
다음에 무슨 일이 일어날지 예측해 보세요. 따라서 생성 언어 모델은 패턴 일치 시스템입니다. 그들은 패턴에 대해 배웁니다.
귀하가 제공한 데이터를 기반으로 합니다. 여기에 예가 있습니다. 배운 내용을 바탕으로
훈련 데이터를 통해 이 문장을 완성하는 방법에 대한 예측을 제공합니다. 저는 샌드위치를 ​​만들고 있습니다.
땅콩버터와 젤리가 들어있습니다. 다음은 Bard를 사용한 동일한 예입니다.
엄청난 양의 텍스트 데이터를 가지고 있으며 인간과 유사한 텍스트를 전달하고 생성할 수 있습니다.
다양한 프롬프트와 질문에 응답합니다. 여기 또 다른 예가 있습니다. 그 의미
인생은... 그리고 Bart는 상황에 맞는 답변을 제공한 다음 가장 높은 확률 응답을 보여줍니다.

생성 AI의 힘은 변압기의 사용에서 비롯됩니다.
Transformers는 2018년 자연어 처리 분야에 혁명을 일으켰습니다.
상위 레벨에서 변환기 모델은 인코더와 디코더로 구성됩니다.
인코더는 입력 시퀀스를 인코딩하고 이를 디코더에 전달합니다. 디코더는 관련 작업에 대한 표현을 디코딩하는 방법을 학습합니다.
변환기에서 환각은 종종 무의미하거나 문법적으로 잘못된 모델에 의해 생성된 단어 또는 문구입니다.
환각은 모델이 충분한 데이터에 대해 훈련되지 않았거나 모델이
시끄러운 데이터나 더러운 데이터에 대해 교육을 받았거나, 모델에 충분한 컨텍스트가 제공되지 않았거나, 모델에 충분한 제약이 주어지지 않았습니다.
환각은 출력 텍스트를 이해하기 어렵게 만들 수 있기 때문에 변환기에 문제가 될 수 있습니다.
또한 모델이 부정확하거나 오해의 소지가 있는 정보를 생성할 가능성을 높일 수도 있습니다.
프롬프트는 대규모 언어 모델에 입력으로 제공되는 짧은 텍스트입니다.
그리고 다양한 방법으로 모델의 출력을 제어하는 ​​데 사용될 수 있습니다.
프롬프트 디자인은 대규모 언어 모델에서 원하는 출력을 생성하는 프롬프트를 만드는 프로세스입니다.
이전에 언급했듯이 Gen AI는 입력한 교육 데이터에 많이 의존합니다.
그리고 입력된 데이터의 패턴과 구조를 분석하여 학습합니다.
그러나 브라우저 기반 프롬프트에 액세스하면 사용자는 자신만의 콘텐츠를 생성할 수 있습니다.
데이터를 기반으로 한 입력 유형에 대한 그림을 보여 드렸습니다.
연관된 모델 유형은 다음과 같습니다.

텍스트-텍스트.
텍스트-텍스트 모델은 자연어 입력을 받아 텍스트 출력을 생성합니다.
이러한 모델은 예를 들어 한 언어에서 다른 언어로의 번역과 같이 텍스트 쌍 간의 매핑을 학습하도록 훈련되었습니다.
텍스트를 이미지로.
텍스트-이미지 모델은 각각 짧은 텍스트 설명이 포함된 대규모 이미지 세트에 대해 학습됩니다.
확산은 이를 달성하는 데 사용되는 한 가지 방법입니다.
텍스트를 비디오로, 텍스트를 3D로.
텍스트-비디오 모델은 텍스트 입력으로부터 비디오 표현을 생성하는 것을 목표로 합니다.
입력 텍스트는 단일 문장부터 전체 스크립트까지 무엇이든 될 수 있습니다.
그리고 출력은 입력된 텍스트에 해당하는 비디오입니다.
마찬가지로, 텍스트-3D 모델은 사용자의 텍스트 설명에 해당하는 3차원 개체를 생성합니다.
예를 들어 게임이나 다른 3D 세계에서 사용할 수 있습니다.
텍스트-작업.
텍스트-작업 모델은 텍스트 입력을 기반으로 정의된 작업이나 작업을 수행하도록 학습됩니다.
이 작업은 질문에 답하기, 검색 수행하기, 예측하기, 일종의 조치 취하기 등 광범위한 조치일 수 있습니다.
예를 들어, 텍스트-작업 모델은 웹 UI를 탐색하거나 GUI를 통해 문서를 변경하도록 훈련될 수 있습니다.
기초 모델은 적응하도록 설계된 방대한 양의 데이터에 대해 사전 훈련된 대규모 AI 모델입니다.
또는 감정 분석, 이미지 캡션 작성 및 개체 인식과 같은 광범위한 다운스트림 작업에 맞게 미세 조정됩니다.
기초 모델은 의료, 금융, 고객 서비스를 포함한 많은 산업에 혁명을 일으킬 수 있는 잠재력을 가지고 있습니다.
이는 사기를 탐지하고 맞춤형 고객 지원을 제공하는 데 사용될 수 있습니다.
Vertex AI는 기초 모델이 포함된 모델 정원을 제공합니다.
언어 기반 모델에는 채팅 및 텍스트용 PaLM API가 포함되어 있습니다.
비전 기초 모델에는 안정적인 확산이 포함되어 있으며 이는 텍스트 설명에서 고품질 이미지를 생성하는 데 효과적인 것으로 나타났습니다.
고객이 제품이나 서비스에 대해 어떻게 느끼는지에 대한 감정을 수집해야 하는 사용 사례가 있다고 가정해 보겠습니다.
해당 목적으로 분류 작업 감정 분석 작업 모델을 사용할 수 있습니다.
점유 분석을 수행해야 한다면 어떻게 될까요?
사용 사례에 맞는 작업 모델이 있습니다.
여기에는 Gen AI 애플리케이션이 표시됩니다.
상단의 코드 아래 두 번째 블록에 표시된 코드 생성의 예를 살펴보겠습니다.
이 예에서는 Python에서 JSON으로 변환하는 코드 파일 변환 문제를 입력했습니다.
저는 바드를 ​​사용합니다.
그리고 프롬프트 상자에 다음을 삽입합니다.
두 개의 열이 있는 Pandas DataFrame이 있습니다. 하나는 파일 이름이고 다른 하나는 생성된 시간입니다.
화면에 표시된 형식의 JSON 파일로 변환하려고 합니다.
Bard는 이 작업을 수행하는 데 필요한 단계와 코드 조각을 반환합니다.
여기 내 출력은 JSON 형식입니다.
좋아진다.
저는 Colab으로 알려진 Google의 무료 브라우저 기반 Jupyter Notebook을 사용하고 있습니다.
그리고 Python 코드를 Google Colab으로 내보냅니다.
요약하면 Bart 코드 생성은 소스 코드 줄을 디버깅하고 코드를 한 줄씩 설명하는 데 도움이 될 수 있습니다.
라인, 데이터베이스에 대한 SQL 쿼리 작성, 코드를 한 언어에서 다른 언어로 번역하고 소스 코드에 대한 문서 및 튜토리얼을 생성합니다.
Generative AI Studio를 사용하면 Google Cloud의 애플리케이션에서 활용할 수 있는 Gen AI 모델을 빠르게 탐색하고 맞춤설정할 수 있습니다.
Generative AI Studio는 개발자가 쉽게 시작할 수 있도록 다양한 도구와 리소스를 제공하여 Gen AI 모델을 생성하고 배포하는 데 도움을 줍니다.
예를 들어 사전 학습된 모델 라이브러리가 있습니다.
모델을 미세 조정하는 도구가 있습니다.
모델을 프로덕션에 배포하기 위한 도구가 있습니다.
그리고 개발자들이 아이디어를 공유하고 협업할 수 있는 커뮤니티 포럼도 있습니다.
Generative AI App Builder를 사용하면 코드를 작성할 필요 없이 Gen AI 앱을 만들 수 있습니다.
Gen AI App Builder에는 앱을 쉽게 디자인하고 구축할 수 있는 드래그 앤 드롭 인터페이스가 있습니다.
앱 콘텐츠를 쉽게 생성하고 편집할 수 있는 시각적 편집기가 있습니다.
사용자가 앱 내에서 정보를 검색할 수 있는 검색 엔진이 내장되어 있습니다.
그리고 사용자가 자연어를 사용하여 앱과 상호 작용할 수 있도록 돕는 대화형 AI 엔진이 있습니다.
자신만의 디지털 도우미, 사용자 정의 검색 엔진, 지식 기반, 교육 애플리케이션 등을 만들 수 있습니다.
PaLM API를 사용하면 Google의 대규모 언어 모델과 Gen AI 도구를 테스트하고 실험할 수 있습니다.
프로토타입 제작을 빠르고 쉽게 만들기 위해 개발자는 PaLM API를 Maker 제품군과 통합하고 이를 사용하여 그래픽 사용자 인터페이스를 통해 API에 액세스할 수 있습니다.
이 제품군에는 모델 교육 도구, 모델 배포 도구 및 모델 모니터링 도구와 같은 다양한 도구가 포함되어 있습니다.
모델 훈련 도구는 개발자가 다양한 알고리즘을 사용하여 데이터에 대해 ML 모델을 훈련하는 데 도움이 됩니다.
모델 배포 도구는 개발자가 다양한 배포 옵션을 사용하여 ML 모델을 프로덕션에 배포하는 데 도움이 됩니다.
모델 모니터링 도구는 개발자가 대시보드와 다양한 측정항목을 사용하여 프로덕션에서 ML 모델의 성능을 모니터링하는 데 도움이 됩니다.

