

All Readings: Introduction to Large Language Models (G-LLM-I)
Here are the assembled readings on large language models:
● Introduction to Large Language Models
https://developers.google.com/machine-learning/resources/intro-llms
● Language Models are Few-Shot Learners:
https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64aPaper.pdf
● Getting Started with LangChain + Vertex AI PaLM API
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/orchestration/langchain/intro_langchain_palm_api.ipynb
● Learn about LLMs, PaLM models, and Vertex AI
https://cloud.google.com/vertex-ai/docs/generative-ai/learn-resources
● Building AI-powered apps on Google Cloud databases using pgvector, LLMs and LangChain
https://cloud.google.com/blog/products/databases/using-pgvector-llms-and-langchainwith-google-cloud-databases
● Training Large Language Models on Google Cloud
https://github.com/GoogleCloudPlatform/llm-pipeline-examples
● Prompt Engineering for Generative AI
https://developers.google.com/machine-learning/resources/prompt-eng
● PaLM-E: An embodied multimodal language model:
https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html
● Parameter-efficient fine-tuning of large-scale pre-trained language models
https://www.nature.com/articles/s42256-023-00626-4
● Understanding Parameter-Efficient LLM Finetuning: Prompt Tuning And Prefix Tuning
● Parameter-Efficient Fine-Tuning of Large Language Models with LoRA and QLoRA
https://www.analyticsvidhya.com/blog/2023/08/lora-and-qlora/
● Solving a machine-learning mystery:
https://news.mit.edu/2023/large-language-models-in-context-learning-0207



Here are the assembled readings on generative AI:
● Background: What is a Generative Model?
https://developers.google.com/machine-learning/gan/generative
● Gen AI for Developers
https://cloud.google.com/ai/generative-ai#section-3

● Ask a Techspert: What is generative AI?
https://blog.google/inside-google/googlers/ask-a-techspert/what-is-generative-ai/

● What is generative AI?
https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-generative-ai

Building the most open and innovative AI ecosystem:
https://cloud.google.com/blog/products/ai-machine-learning/building-an-open-generative-ai-partner-ecosystem
● Generative AI is here. Who Should Control It?
https://www.nytimes.com/2022/10/21/podcasts/hard-fork-generative-artificial-intelligence.html
● Stanford U & Google’s Generative Agents Produce Believable Proxies of Human
Behaviors:
https://syncedreview.com/2023/04/12/stanford-u-googles-generative-agents-produce-believable-proxies-of-human-behaviours/
● Generative AI: Perspectives from Stanford HAI:
https://hai.stanford.edu/sites/default/files/2023-03/Generative_AI_HAI_Perspectives.pdf
● Generative AI at Work:
https://www.nber.org/system/files/working_papers/w31161/w31161.pdf
● The future of generative AI is niche, not generalized:
https://www.technologyreview.com/2023/04/27/1072102/the-future-of-generative-ai-isniche-not-generalized/
● The implications of Generative AI for businesses:
https://www2.deloitte.com/us/en/pages/consulting/articles/generative-artificial-intelligence.html
● Proactive Risk Management in Generative AI:
https://www2.deloitte.com/us/en/pages/consulting/articles/responsible-useof-generative-ai.html
● How Generative AI Is Changing Creative Work:
https://hbr.org/2022/11/how-generative-ai-is-changing-creative-work

Additional Resources:
● Attention is All You Need: https://research.google/pubs/pub46201/
● Transformer: A Novel Neural Network Architecture for Language Understanding:
https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html
● Transformer on Wikipedia:
https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#:~:text=Transformers%20were%20introduced%20in%202017allowing%20training%20on%20larger%20datasets.

● What is Temperature in NLP? https://lukesalamone.github.io/posts/what-is-temperature/
● Model Garden: https://cloud.google.com/model-garden
● Auto-generated Summaries in Google Docs:
https://ai.googleblog.com/2022/03/auto-generated-summaries-in-google-docs.html


LLM

LLM(대형 언어 모델) 정의 LLM 사용 사례 설명 프롬프트 조정 설명

Google의 Gen AI 개발 도구 설명 LLM(대형 언어 모델)은 딥 러닝의 하위 집합입니다.
딥 러닝에 대해 자세히 알아보려면 Generative AI 소개 과정 비디오를 참조하세요.
LLM과 Generative AI는 교차하며 둘 다 딥 러닝의 일부입니다.

여러분이 많이 듣게 될 AI의 또 다른 영역은 생성 AI입니다.
텍스트 이미지, 오디오, 합성 데이터 등 새로운 콘텐츠를 생산할 수 있는 일종의 인공지능이다.
그렇다면 대규모 언어 모델이란 무엇입니까?
대규모 언어 모델은 사전 훈련된 다음 특정 목적에 맞게 미세 조정할 수 있는 대규모 범용 언어 모델을 의미합니다.

사전 학습 및 미세 조정은 무엇을 의미하나요?
개를 훈련시킨다고 상상해 보세요.
종종 당신은 앉기, 와서, 아래로, 머물기와 같은 개 기본 명령을 훈련시킵니다.
이러한 명령은 일반적으로 일상 생활에 충분하며 개가 좋은 개 시민이 되는 데 도움이 됩니다.
다만, 경찰견, 안내견, 사냥개 등 특수견이 필요한 경우에는 특수훈련을 추가합니다.
이와 유사한 아이디어는 대규모 언어 모델에도 적용됩니다.

이러한 모델은 산업 전반에 걸쳐 텍스트 분류, 질문 답변, 문서 요약 및 텍스트 생성과 같은 공통 언어 문제를 해결하기 위한 일반적인 목적으로 훈련되었습니다.
그런 다음 상대적으로 작은 규모의 현장 데이터 세트를 사용하여 소매, 금융, 엔터테인먼트 등 다양한 분야의 특정 문제를 해결하도록 모델을 맞춤화할 수 있습니다.

개념을 대규모 언어 모델의 세 가지 주요 기능으로 더 세분화해 보겠습니다.

크다는 두 가지 의미를 나타냅니다. 첫 번째는 훈련 데이터 세트의 엄청난 크기(때로는 페타바이트 규모)입니다.
두 번째는 매개변수 수를 나타냅니다(ML에서는 매개변수를 종종 하이퍼매개변수라고 함).
매개변수는 기본적으로 기계가 모델 훈련을 통해 학습한 기억과 지식입니다.
매개변수는 텍스트 예측과 같은 문제 해결에서 모델의 기술을 정의합니다.
범용이란 모델이 일반적인 문제를 해결하기에 충분하다는 것을 의미합니다.

이런 생각을 갖게 된 데에는 두 가지 이유가 있습니다. 
첫째는 특정 작업에 관계없이 인간 언어의 공통성이며, 둘째는 자원 제한입니다.
특정 조직만이 거대한 데이터 세트와 엄청난 수의 매개변수를 사용하여 이러한 대규모 언어 모델을 훈련할 수 있는 능력을 갖추고 있습니다.
다른 사람들이 사용할 수 있는 기본적인 언어 모델을 만들도록 하는 것은 어떻습니까?
이는 사전 훈련 및 미세 조정이라는 마지막 지점으로 이어집니다. 이는 대규모 언어 모델을 사전 훈련하는 것을 의미합니다.
큰 데이터 세트를 사용하여 일반 목적으로 조정한 다음 훨씬 더 작은 데이터 세트를 사용하여 특정 목적에 맞게 미세 조정합니다.

대규모 언어 모델 사용의 이점은 간단합니다. 

첫째, 단일 모델을 다양한 작업에 사용할 수 있습니다.
이것은 꿈이 이루어진 것입니다.
페타바이트 규모의 데이터로 훈련되고 수십억 개의 매개변수를 생성하는 이러한 대규모 언어 모델은
언어 번역, 문장 완성, 텍스트 분류, 질문 답변 등 다양한 작업을 해결할 수 있을 만큼 똑똑합니다.

둘째, 대규모 언어 모델은 특정 문제를 해결하기 위해 맞춤화할 때 최소한의 현장 학습 데이터가 필요합니다.
대규모 언어 모델은 도메인 훈련 데이터가 거의 없어도 적절한 성능을 얻습니다.
즉, 퓨샷 또는 제로샷 시나리오에 사용할 수 있습니다.
머신러닝에서 '퓨샷(few-shot)'은 최소한의 데이터로 모델을 훈련하는 것을 의미하고, '제로샷(zero-shot)'은
모델은 이전 훈련에서 명시적으로 가르치지 않은 내용을 인식할 수 있습니다.

셋째, 더 많은 데이터와 매개변수를 추가하면 대규모 언어 모델의 성능이 지속적으로 향상됩니다.
PaLM을 예로 들어보겠습니다.
2022년 4월, Google은 여러 언어 작업에서 최첨단 성능을 달성하는 5,400억 개의 매개변수 모델인 PaLM(Pathways Language Model의 약자)을 출시했습니다.
PaLM은 밀도가 높은 디코더 전용 Transformer 모델입니다.
5400억 개의 매개변수가 있습니다.
이는 Google이 여러 TPU v4 Pod에서 단일 모델을 효율적으로 학습할 수 있도록 지원하는 새로운 Pathways 시스템을 활용합니다.
Pathway는 많은 작업을 한 번에 처리하고, 새로운 작업을 빠르게 학습하며, 세상에 대한 더 나은 이해를 반영하는 새로운 AI 아키텍처입니다.
이 시스템을 통해 PaLM은 가속기에 대한 분산 계산을 조정할 수 있습니다.
이전에 PaLM이 트랜스포머 모델이라고 언급했습니다.
Transformer 모델은 인코더와 디코더로 구성됩니다.
인코더는 입력 시퀀스를 인코딩하고 이를 디코더에 전달합니다. 디코더는 관련 작업에 대한 표현을 디코딩하는 방법을 학습합니다.
우리는 전통적인 프로그래밍에서 신경망, 생성 모델에 이르기까지 먼 길을 걸어왔습니다!

전통적인 프로그래밍에서는 고양이를 구별하기 위한 규칙(유형: 동물, 다리: 4, 귀: 2, 모피: 예, 좋아하는 것: 털실, 개박하)을 하드 코딩해야 했습니다.
신경망의 물결 속에서 우리는 네트워크에 고양이와 개의 사진을 제공하고 "이것이 고양이인가요?"라고 질문할 수 있으며, 그러면 고양이를 예측할 수 있습니다.
생성의 물결에서 우리는 사용자로서 텍스트, 이미지, 오디오, 비디오 등 자체 콘텐츠를 생성할 수 있습니다.
예를 들어 PaLM(또는 Pathways Language Model 또는 LaMDA(또는 대화 응용 프로그램을 위한 언어 모델))과 같은 모델은 전 세계의 여러 소스에서 매우 큰 데이터를 수집합니다.
인터넷을 사용하여 프롬프트에 입력하든, 프롬프트에 구두로 말하든 질문을 함으로써 간단히 사용할 수 있는 기초 언어 모델을 구축합니다.
사용자로서 우리는 이러한 언어 모델을 사용하여 무엇보다도 텍스트를 생성하거나 질문에 답하거나 데이터를 요약할 수 있습니다.
따라서 "고양이가 무엇인지"라고 물으면 고양이에 대해 배운 모든 정보를 제공할 수 있습니다.
사전 학습된 모델을 사용한 LLM 개발과 기존 ML 개발을 비교해 보겠습니다.

첫째, LLM 개발을 사용하면 전문가가 될 필요가 없습니다.
훈련 예제가 필요하지 않으며 모델을 훈련할 필요도 없습니다.
당신이 해야 할 일은 명확하고 간결하며 유익한 프롬프트를 만드는 과정인 프롬프트 디자인에 대해 생각하는 것뿐입니다.
자연어 처리(NLP)의 중요한 부분입니다.
기존 기계 학습에서는 전문 지식, 교육 예제, 모델 교육, 시간 및 하드웨어 계산이 필요합니다.
텍스트 생성 사용 사례의 예를 살펴보겠습니다.
질문 응답(QA)은 자연어로 제기된 질문에 자동으로 대답하는 작업을 다루는 자연어 처리의 하위 분야입니다.
QA 시스템은 일반적으로 많은 양의 텍스트와 코드에 대해 교육을 받고 사실, 정의, 의견 기반 질문을 포함한 광범위한 질문에 답할 수 있습니다.
여기서 핵심은 이러한 질문 응답 모델을 개발하려면 도메인 지식이 필요하다는 것입니다.
예를 들어, 고객 IT 지원, 의료 또는 공급망에 대한 질문 응답 모델을 개발하려면 도메인 지식이 필요합니다.
생성적 QA를 사용하여 모델은 컨텍스트를 기반으로 직접 자유 텍스트를 생성합니다.
도메인 지식이 필요하지 않습니다.
Google AI가 개발한 대규모 언어 모델 챗봇인 Bard에게 주어진 세 가지 질문을 살펴보겠습니다.

질문 1 - 올해 매출은 10만 달러입니다.
비용은 60,000달러입니다.
순이익은 얼마입니까?
Bard는 먼저 순이익 계산 방법을 공유한 후 계산을 수행합니다.
그런 다음 Bard는 순이익의 정의를 제공합니다.
또 다른 질문이 있습니다. 보유 재고가 6,000개입니다.
신규 주문에는 8,000개 단위가 필요합니다.
주문을 완료하려면 몇 단위를 채워야 합니까?
이번에도 Bard는 계산을 수행하여 질문에 답합니다.

마지막 예는 10개 지역에 1,000개의 센서가 있습니다.
각 지역에는 평균 몇 개의 센서가 있습니까?

Bard는 문제 해결 방법에 대한 예와 몇 가지 추가 컨텍스트를 통해 질문에 답합니다.
내 질문 각각에서 원하는 답변을 얻었습니다.
이는 프롬프트 디자인(Prompt Design) 때문이다.
신속한 설계와 신속한 엔지니어링은 자연어 처리에서 밀접하게 관련된 두 가지 개념입니다.
두 가지 모두 명확하고 간결하며 유익한 프롬프트를 만드는 과정을 포함합니다.

그러나 둘 사이에는 몇 가지 주요 차이점이 있습니다.
프롬프트 디자인은 시스템이 수행하도록 요청받은 특정 작업에 맞게 조정된 프롬프트를 만드는 프로세스입니다.
예를 들어, 시스템이 영어에서 프랑스어로 텍스트를 번역하도록 요청받는 경우,
프롬프트는 영어로 작성해야 하며 번역이 프랑스어로 되어야 함을 지정해야 합니다.
프롬프트 엔지니어링은 성능을 향상시키도록 설계된 프롬프트를 생성하는 프로세스입니다.
여기에는 도메인별 지식을 사용하거나, 원하는 결과의 예를 제공하거나, 특정 시스템에 효과적인 것으로 알려진 키워드를 사용하는 것이 포함될 수 있습니다.
일반적으로 프롬프트 설계는 보다 일반적인 개념인 반면, 프롬프트 엔지니어링은 보다 전문적인 개념입니다.
신속한 설계는 필수적인 반면, 신속한 엔지니어링은 높은 수준의 정확성이나 성능이 요구되는 시스템에만 필요합니다.
대형 언어 모델에는 일반 언어 모델, 명령 조정 및 대화 조정의 세 가지 종류가 있습니다.

각각은 다른 방식으로 메시지를 표시해야 합니다.
일반 언어 모델은 훈련 데이터의 언어를 기반으로 다음 단어(기술적으로 토큰)를 예측합니다.
이것은 일반 언어 모델의 예입니다. 다음 단어는 훈련 데이터의 언어를 기반으로 하는 토큰입니다.
이 예에서 고양이는 앉았습니다. 다음 단어는 "the"여야 하며 "the"가 다음 단어일 가능성이 가장 높다는 것을 알 수 있습니다.
이 유형을 검색의 '자동 완성'이라고 생각하세요.
조정된 명령에서 모델은 입력에 제공된 명령에 대한 응답을 예측하도록 훈련됩니다.
예를 들어, "x"라는 텍스트를 요약하고, 'x" 스타일로 시를 생성하고, 키워드 목록을 기반으로 제공합니다.
"x"에 대한 의미론적 유사성에 대해. 그리고 이 예에서는 텍스트를 중립, 부정 또는 긍정적으로 분류합니다. Dialog tuned에서 모델은 다음과 같습니다.
다음 응답으로 대화를 갖도록 훈련되었습니다. 대화 조정 모델은 요청이 일반적으로 조정되는 특수한 명령 사례입니다.
채팅 봇에 대한 질문으로 구성됩니다. 대화 조정은 더 긴 대화의 맥락에서 이루어질 것으로 예상됩니다.
일반적으로 자연스러운 질문과 같은 표현에 더 잘 작동합니다. 일련의 사고 추론은 모델이 더 나은 결과를 얻는다는 관찰입니다.
정답의 이유를 설명하는 텍스트를 처음 출력할 때 정답입니다. 질문을 살펴보겠습니다. Roger는 테니스 5개를 가지고 있습니다.
그는 테니스 공 2캔을 더 구입합니다. 각 캔에는 테니스 공 3개가 들어 있습니다. 그는 테니스 공을 몇 개 가지고 있습니까?
지금? 이 질문은 처음에는 아무런 응답 없이 제기되었습니다. 모델이 정답을 직접 얻을 가능성은 적습니다. 그러나

두 번째 질문을 하면 출력이 정답으로 끝날 가능성이 더 높습니다. 할 수 있는 모델
모든 것에는 실질적인 한계가 있습니다. 작업별 튜닝을 통해 LLM의 안정성을 높일 수 있습니다. Vertex AI는 작업별 기반 모델을 제공합니다. 당신이 가지고 있다고 가정 해 봅시다
감정(또는 고객이 제품이나 서비스에 대해 어떻게 느끼는지)을 수집해야 하는 사용 사례에서는 다음을 사용할 수 있습니다.
분류 작업 감정 분석 작업 모델. 비전 작업에도 동일합니다. 점유 분석을 수행해야 하는 경우 작업이 있습니다.
사용 사례에 맞는 특정 모델. 모델을 튜닝하면 작업 예시를 기반으로 모델 응답을 사용자 정의할 수 있습니다.
모델이 수행하기를 원하는 것입니다. 이는 본질적으로 모델을 새로운 영역이나 집합에 적용하는 프로세스입니다.
새로운 데이터로 모델을 훈련하여 맞춤형 사용 사례를 제공합니다. 예를 들어 훈련 데이터를 수집하고 모델을 구체적으로 "조정"할 수 있습니다.
법률 또는 의료 분야의 경우. 자체 데이터세트를 가져와 재교육하는 '미세 조정'을 통해 모델을 추가로 조정할 수도 있습니다.
LLM의 모든 가중치를 조정하여 모델을 만듭니다. 이를 위해서는 대규모 교육 작업과 자체적으로 미세 조정된 모델을 호스팅해야 합니다. 

여기 의료 데이터에 대해 훈련된 의료 기반 모델의 예입니다. 질의응답, 영상분석, 유사환자 찾기 등의 업무를 수행합니다.
미세 조정은 비용이 많이 들고 많은 경우 현실적이지 않습니다. 그렇다면 보다 효율적인 튜닝 방법이 있을까요? 예. 매개변수 효율적인 튜닝 방법은 다음과 같습니다.
모델을 복제하지 않고 자신의 사용자 정의 데이터로 대규모 언어 모델을 조정하는 방법입니다. 기본 모델 자체는 변경되지 않습니다.
대신, 추론 시 교체할 수 있는 소수의 추가 레이어가 조정됩니다. 생성 AI 스튜디오
Google Cloud의 애플리케이션에서 활용할 수 있는 생성적 AI 모델을 빠르게 탐색하고 맞춤설정할 수 있습니다. Generative AI Studio가 도움이 됩니다.
개발자는 쉽게 시작할 수 있는 다양한 도구와 리소스를 제공하여 생성적 AI 모델을 만들고 배포합니다.
예를 들면 다음과 같습니다. 사전 훈련된 모델 라이브러리 모델 미세 조정을 위한 도구 모델을 프로덕션에 배포하기 위한 도구 개발자를 위한 커뮤니티 포럼
아이디어를 공유하고 협업할 수 있습니다. Vertex AI Search and Conversation(이전의 Vertex AI Search and Conversation)을 사용하여 고객과 직원을 위한 생성적 AI 검색 및 대화를 구축하세요.
젠 앱 빌더). 코딩이 거의 또는 전혀 없고 사전 기계 학습 경험 없이 구축할 수 있습니다. 나만의 것을 만들 수 있습니다: 챗봇, 디지털 보조자,
사용자 정의 검색 엔진, 기술 자료, 교육 애플리케이션 등. PaLM API를 사용하면 Google의 대규모 언어 모델을 테스트하고 실험할 수 있으며
Gen AI 도구. 프로토타입 제작을 더 빠르고 쉽게 만들기 위해 개발자는 PaLM API를 MakerSuite와 통합하고 이를 사용하여
그래픽 사용자 인터페이스를 사용하는 API입니다. 이 제품군에는 모델 교육 도구, 모델과 같은 다양한 도구가 포함되어 있습니다.
배포 도구 및 모델 모니터링 도구입니다. 모델 훈련 도구는 개발자가 다양한 알고리즘을 사용하여 데이터에 대해 ML 모델을 훈련하는 데 도움이 됩니다. 그만큼
모델 배포 도구는 개발자가 다양한 배포 옵션을 사용하여 ML 모델을 프로덕션에 배포하는 데 도움이 됩니다. 모델 모니터링 도구가 도움이 됩니다.
개발자는 대시보드와 다양한 측정항목을 사용하여 프로덕션에서 ML 모델의 성능을 모니터링합니다. 지금은 여기까지입니다.

