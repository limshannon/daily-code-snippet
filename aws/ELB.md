from: https://explore.skillbuilder.aws/learn/course/10985/play/40310/working-with-elastic-load-balancing-korean

## ELB ##
* Amazon EC2에서 여러 서버 웹 팜을 시작
* 부트스트래핑 기술을 사용하여 Amazon Simple Storage Service(S3)에서 다운로드한 Apache, PHP 및 간단한 PHP 애플리케이션으로 Linux 인스턴스를 구성
* Amazon EC2 웹 서버 인스턴스 앞에 배치되는 로드 밸런서를 생성 및 구성
* 로드 밸런서에 대한 Amazon CloudWatch 지표를 확인

 Elastic Load Balancing을 사용하여 단일 가용 영역에서 여러 Amazon Elastic Compute Cloud(EC2) 인스턴스에 걸쳐 트래픽을 로드 밸런스합니다. 여러 Amazon EC2 인스턴스에 간단한 애플리케이션을 배포하고 브라우저에서 이 애플리케이션을 보면서 로드 밸런싱을 관찰합니다.

먼저, 한 쌍의 인스턴스를 시작하고 이들을 부트스트랩하여 웹 서버와 콘텐츠를 설치한 다음 Amazon EC2 DNS 레코드를 사용하여 독립적으로 인스턴스에 액세스합니다. 다음으로 Elastic Load Balancing을 설정하고 인스턴스를 로드 밸런서에 추가한 다음 DNS 레코드에 다시 액세스하여 서버 간 요청 로드 밸런스를 관찰합니다. 마지막으로 Amazon CloudWatch에서 Elastic Load Balancing 지표를 확인합니다.

* Elastic Load Balancing은 수신되는 애플리케이션 트래픽을 여러 Amazon EC2 인스턴스에 자동으로 분산합니다. 따라서 애플리케이션의 내결함성 수준을 크게 높이고, 애플리케이션 트래픽을 분산하는 데 필요한 로드 밸런싱 용량을 원활하게 제공할 수 있습니다.

* Elastic Load Balancing을 사용해 여러 인스턴스 및 여러 가용 영역에서 트래픽을 자동으로 라우팅하면 애플리케이션의 내결함성 수준을 크게 높일 수 있습니다. Elastic Load Balancing은 비정상 인스턴스를 감지하고 트래픽을 나머지 정상 인스턴스로 다시 라우팅하여 정상적인 Amazon EC2 인스턴스만 트래픽을 수신하도록 합니다. 한 가용 영역의 모든 Amazon EC2 인스턴스가 비정상적이지만 여러 가용 영역에 Amazon EC2 인스턴스를 설정해두었다면 Elastic Load Balancing이 다른 영역에 있는 정상적인 Amazon EC2 인스턴스로 트래픽을 라우팅합니다.

* Elastic Load Balancing은 애플리케이션 트래픽의 수요에 부합하도록 요청 처리량을 자동 조정합니다. 또한 Elastic Load Balancing은 Auto Scaling과 통합하여 수동적인 개입 없이 다양한 수준의 트래픽을 처리할 수 있는 백엔드 용량을 보장합니다.

* Elastic Load Balancing은 Amazon Virtual Private Cloud(VPC)와 연동하여 강력한 네트워킹 및 보안 기능을 제공합니다. 내부 로드 밸런서(인터넷 연결 안 됨)를 생성하여 가상 네트워크 내에서 프라이빗 IP 주소를 사용해 트래픽을 라우팅할 수 있습니다. 내부 및 인터넷용 로드 밸런서를 사용하여 애플리케이션 티어 간에 발생하는 트래픽을 라우팅하여 멀티 티어 아키텍처를 구현할 수 있습니다. 멀티 티어 아키텍처를 사용하면 애플리케이션 인프라에 프라이빗 IP 주소와 보안 그룹을 사용할 수 있으므로 퍼블릭 IP 주소가 있는 인터넷 연결 티어만 노출할 수 있습니다.

* Elastic Load Balancing은 통합 인증 관리 및 SSL 복호화를 지원하므로 로드 밸런서의 SSL 설정을 중앙 집중식으로 관리하고 인스턴스로부터 CPU 집중 사용 작업을 오프로드할 수 있습니다.

이 실습 가이드에서는 Elastic Load Balancing의 기본 개념을 단계별로 설명합니다. 그러나 Elastic Load Balancing 개념에 대한 간략한 개요만 제공할 수 있습니다. 자세한 내용은 http://aws.amazon.com/elasticloadbalancing/을 참조하십시오.

