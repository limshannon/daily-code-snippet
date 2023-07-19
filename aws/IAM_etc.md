### SDK ###
AWS SDK를 살펴보겠습니다 SDK란 무엇일까요?
만약 지금까지 사용했던 CLI를 사용하지 않고
애플리케이션 코드에서 직접
AWS 작업을 하려면 어떻게 해야 할까요?
이때 소프트웨어 개발 키트인 SDK를 사용합니다
그리고 다양한 언어의 AWS용 공식 SDK가 있습니다
Java, .NET와 Node.js가 있고
PHP, Python, Go, Ruby 그리고 C++ 등이 있습니다. 그리고 이 목록은 계속 늘어날 것입니다
CLI를 사용할 때는 Python SDK를 사용했는데요
CLI가 Phython 언어를 사용하고 Boto3 SDK를 사용하기 때문입니다
그래서 SDK는 DynamoDB나 Amazon S3와 같은 Amazon 서비스에서 API 호출을 발행할 때 사용합니다
하지만 CLI도 Phython SDK (boto3)를 사용합니다 그래서 언제 SDK를 쓰는지가 시험에 출제됩니다
Lamda 함수를 다룰 때 SDK를 살펴보고 SDK의 원리를 코드를 통해 연습해 보겠습니다

- 리전을 지정하지 않고 기본 리전을 구성하지 않으면 API 호출을 위해서 SDK에서 us-east-1이 기본으로 선택된다는 것입니다

### AWS Limit ###
이제 할당량이라고도 부르는 AWS 제한에 관해 살펴봅니다

제한에는 2가지가 있는데요 
- 첫 번째는 API 비율 제한으로 AWS API를 연속으로 호출하는 횟수를 말합니다.
예를 들어, Amazon EC2의 DescribeInstances API는 초당 100회 호출의 제한이 있습니다
그리고 Amazon S3의 GetObject는 접두부와 초당 5,500GET의 제한이 있습니다
그래서 진행하다 보면 간헐적 오류도 발생하는데 제한되기 때문입니다. 
그래서 지수 백오프 전략을 사용해야 하는데 다음 슬라이드에서 설명하겠습니다.
이 경우에는 이런 오류가 계속 발생하는데요 애플리케이션 사용량이 많아서 계속 제한을 초과하는 경우에는 API 제한 증가를 요청해 계속 사용하도록 합니다.
DescribeInstances의 제한을 초당 100회 호출 이상 또는 300회 이상으로 요청할 수도 있죠. AWS에 문의해야 하죠.

실행할 수 있는 리소스 수를 말합니다
온디맨드 표준 인스턴스의 경우 최대 1,152개의 가상 CPU를 사용할 수 있으며 계정에서 더 많은 가상 CPU를 사용하려면 티켓을 오픈해서 서비스 한도 증가를 요청할 수 있습니다

서비스 할당량을 늘리려면 서비스 할당량 API를 이용해서 프로그램에서 요청할 수 있습니다
리소스에 API 비율 제한과 서비스 할당량이 있는 것이죠 
그리고 말씀드린 대로 간헐적 오류 발생 시에는 지수 백오프를 사용해야 합니다
지수 백오프는 언제 사용할까요? 조절 오류(ThrottlingException)가 발생한 경우입니다

이 질문도 자주 출제되는데 조절 오류는 API 호출을 많이 했기 때문이며 일반적으로 그 답은 지수 백오프입니다
그래서 AWS SDK를 사용하시면 이 재시도 메커니즘은 이미 SDK 동작에 포함되어 있습니다
하지만 AWS API를 그대로 사용하신다면 지수 백오프를 실행해야 합니다
그래서 지수 백오프를 재시도 해야 하는 경우를 고르는 문제가 시험에 자주 출제됩니다
자체 SDK를 실행하는 경우나 자체 사용자 정의 HTTP 호출을 사용하는 경우 500으로 시작하는 오류 코드의
서버 오류가 발생하면 재시도 메커니즘을 실행해야 합니다.
503과 같이 5xx로 시작하는 오류입니다 이러한 서버 오류와 조절 오류는 재시도 가능하지만
4xx 클라이언트 오류에서는 재시도 혹은 지수 백오프를 실행해서는 안 됩니다
400 오류의 경우에는 클라이언트를 통해 뭔가 잘못 보내졌다는 의미로
계속 재시도 하면 동일한 오류가 계속 발생합니다 

지수 백오프의 원리는 무엇일까요?
이제 1초 동안 첫 번째 요청을 시도하고 다음 요청까지 대기 시간을 두 배로 늘리겠습니다
그러면 2초가 되겠죠 그리고 다음 시도에서 또 2배 늘립니다 4초가 되면 다시 2배로 늘립니다
다음 시도에서는 8초가 되겠죠?
그리고 그다음 시도에서는 16초가 될 것입니다 이러한 지수 백오프의 개념은 더 시도하고 대기하면서 많은 클라이언트가 동시에 이 작업을 실행하면 
그 결과로 서버의 부하가 점점 줄어들어서 서버에서 가능한 많이 응답할 수 있도록 합니다
여기까지 지수 백오프의 전체 개념입니다

### AWS의 자격 증명 공급자 체인 ###
CLI를 사용하면 이어지는 명령에서 자격 증명을 요청합니다
명령줄 옵션을 찾는 것이죠
그래서 명령줄 옵션에서 리전, 출력값, 프로파일 혹은 액세스 키 ID와 비밀 액세스 키 그리고 세션 토큰을 지정하면 그 어떤 것보다 우선이 됩니다
두 번째로 살펴볼 것은 환경 변수입니다 여기 있는 환경 변수 중 한 가지를 설정하고
명령줄 옵션을 설정하지 않으면 환경 변수를 우선 적용합니다
그리고 AWS configure를 실행할 때 CLI 자격 증명 파일을 살펴보고 CLI 구성 파일을 살펴보는데 동일한 방식으로 설정됩니다
그리고 컨테이너 자격 증명을 살펴봅니다 그래서 ECS 작업의 경우에는 컨테이너 자격 증명을 살펴보죠
아직 ECS는 살펴보지 않았지만 곧 살펴보겠습니다 
그리고 마지막으로 EC2 프로파일을 사용하면 인스턴스 프로파일 자격 증명을 살펴봅니다
그래서 최우선은 명령줄 옵션이고 그다음이 환경 변수이며 마지막 우선순위는 EC2 프로파일 자격 증명이나 ECS 컨테이너 자격 증명입니다
따라서 우선순위가 있는 것이며 이는 특정 상황에서 매우 중요한데 곧 설명해 드리겠습니다
이제 SDK를 살펴보죠 예를 들어, Java SDK는 개념이 유사합니다
최우선시 되는 것은 Java 시스템 속성입니다
그다음은 아주 중요한 환경 변수인데 액세스 키 ID나 비밀 액세스 키 같은 것입니다
Java 시스템 속성을 제외하면 그 무엇보다 최우선시 되죠
그리고 기본 자격 증명 프로파일 파일이 있고 Amazon ECS 컨테이너 자격 증명과 인스턴스 프로파일 자격 증명이 있습니다
여기서 기억해야 할 것은 환경 변수가 여전히 우선시 된다는 것인데 예를 들면, EC2 인스턴스 프로파일 자격 증명입니다

왜 이 부분을 말씀드릴까요? 한 가지 상황을 살펴보죠

꼭 기억하셔야 합니다
EC2 인스턴스에서 애플리케이션을 배포하고 IAM 사용자의 환경 변수로
Amazon S3 API를 호출하는 상황입니다 좋지 않은 예시이지만 이렇게 가정해 보겠습니다
그리고 IAM 사용자는 S3FullAccess 권한을 가집니다
이는 Amazon S3의 모든 버킷에서 원하는 작업은 전부 할 수 있다는 뜻입니다
따라서 배포된 애플리케이션은 하나의 Amazon S3 버킷을 사용합니다
이 강의를 시청해 오셨으니 모범 사례에 따르면
EC2 인스턴스 프로파일에서 생성 및 할당하고 있는
IAM 역할 및 EC2 인스턴스 프로파일을 정의하고 있습니다
그리고 이 역할은 사용하고 있는 애플리케이션에 하나의 S3 버킷에만 액세스하는 최소 권한이 할당됐습니다
모두 강의에서 다룬 것으로 최소 권한을 실행하고 EC2 인스턴스 프로파일을 생성하면 어떤 일이 발생할까요?
인스턴스 프로파일이 EC2 인스턴스에 할당됐다고 해도 모든 S3 버킷에 액세스할 수 있습니다
지금까지 배운 것으로 그 이유를 대답할 수 있나요?

그 이유는 이전에 설정한 환경 변수에 자격 증명 체인이 여전히 우선순위를 부여하기 때문입니다
그래서 이것을 없애려면 환경 변수 설정을 해제하고 자격 증명 체인 우선순위를 살펴보면 결국은 EC2 인스턴스 프로파일과 프로파일에서 나오는 권한을 활용하게 됩니다

자격 증명 모범 사례에서는 절대로 자격 증명을 코드에 저장하면 안 됩니다
정말 나쁜 사례입니다 자격 증명에 관한 모범 사례는 자격 증명 체인에서 계속 이어지게 됩니다
AWS에서 작업을 실행하면 최대한 많이 IAM 역할을 사용해야 한다는 뜻입니다
EC2 인스턴스에는 EC2 인스턴스 역할을 쓰고 ECS 작업에는 ECS 역할을 사용해야 하며 Lamda 함수에는 Lamda 역할을 사용해야 하는 것입니다
그래서 AWS에서는 IAM 역할을 최대한 많이 사용하세요
AWS를 사용하지 않으면 환경 변수를 사용하거나 CLI 구성할 때처럼 명명된 프로파일을 사용해야 합니다 하지만 절대로 코드에 자격 증명을 바로 저장하지 마세요

EC2 인스턴스에 IAM 자격 증명을 삽입하지 마세요.

Q.관리자가 Linux EC2 인스턴스를 시작하고 SSH를 사용할 수 있도록 EC2 키 페어를 제공합니다. EC2 인스턴스에 들어간 후 EC2 인스턴스 ID를 얻으려고 합니다. 이 작업을 수행하는 가장 좋은 방법은 무엇일까요?
A. http://169.254.169.254/latest/meta-data에서 메타데이터를 쿼리한다.

Q. 온프레미스 서버에서 애플리케이션을 실행하고 있습니다. 애플리케이션은 S3 버킷에 대한 API 호출을 수행해야 합니다. 가장 안전한 방법으로 이것을 달성하는 방법은 무엇일까요?
A. 애플리케이션에서 사용할 IAM사용자를 생성한 다음 IAM자격증명을 생성하고 자격증명을 환경변수에 넣는다.

Q. S3 버킷에 저장된 민감한 파일을 가져오기 위해 API를 호출하는 데 필요한 IAM 권한이 있는 IAM 역할을 생성했습니다. 새로 생성된 IAM 역할을 EC2 인스턴스에 연결했으며 EC2 인스턴스 내부에서 이러한 파일을 다운로드할 수 있는지 테스트하려고 합니다. 매개변수 값이 중요하므로 변경하지 않고 테스트를 수행하려면 어떻게 해야 할까요?   
A. IAM정책 시뮬레이터 또는 --dry-run AWS CLI 옵션 사용

Q. IAM 역할이 EC2 인스턴스에 연결되면 IAM 역할 이름과 역할에 연결된 IAM 정책을 모두 검색할 수 있습니다.
A. False
인스턴스 메타데이터 서비스를 사용하여 EC2 인스턴스에 연결된 IAM 역할 이름을 검색할 수 있지만 IAM 정책 자체는 검색할 수 없습니다.

Q. EC2 인스턴스 내부에서 EC2 API 호출을 수행하는 동안 승인 예외 오류 메세지 해독...
A. STS decode-authorization-message API 호출 사용

Q.여러분이 보낸 AWS KMS에 대한 마지막 API 호출은 초당 허용되는 최대 API 호출에 도달했기에 쓰로틀링이 시작되었습니다. 어떻게 해야 할까요?
A. 지수 백오프 전략 사용

Q. MFA로 보호되는 API에 대해 API를 호출하기 전에 .......................................를 사용하여 임시 자격 증명을 가져와야 합니다.
A. STS GetSessionToken

Q. AWS CLI는 여러 위치에 있는 자격 증명을 사용하며 특정 위치는 다른 위치보다 우선시 됩니다. 다음 중 AWS CLI가 자격 증명을 찾는 데 사용하는 위치의 올바른 순서는 무엇일까요?
A. 명령줄 옵션 -> 환경 변수 -> EC2인스턴스 프로파일 

https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#config-settings-and-precedence

* 자격 증명 및 구성 설정은 시스템 또는 사용자 환경 변수, 로컬 AWS 구성 파일과 같은 여러 위치에 있거나 명령줄에서 매개 변수로 명시적으로 선언됩니다. 특정 위치가 다른 위치보다 우선합니다. AWS CLI 자격 증명 및 구성 설정은 다음 순서로 우선 적용됩니다.

 1. 명령줄 옵션 – , 및와 같은 다른 위치의 설정을 재정의 --region합니다.--output--profile
 1. 환경 변수 – 시스템의 환경 변수에 값을 저장할 수 있습니다.
 1. 역할 수임 – 구성을 통해 IAM 역할의 권한을 수임하거나aws sts assume-role명령.
 1. 웹 자격 증명으로 역할 수임 – 구성 또는aws sts assume-role명령.
 1. AWS IAM Identity Center(AWS Single Sign-On의 후속 제품) – IAM Identity Center 자격 증명은config파일에 저장되며 명령을 실행할 때 업데이트됩니다 aws configure sso. 파일Linux 또는 macOS의 경우 또는Windows의 경우 config있습니다 ~/.aws/configC:\Users\USERNAME\.aws\config
 1. 자격 증명 파일 –명령을 실행하면credentials및. 파일Linux 또는 macOS의 경우 또는Windows의 경우있습니다configaws configurecredentials~/.aws/credentialsC:\Users\USERNAME\.aws\credentials
 1. 사용자 지정 프로세스 – 외부 소스에서 자격 증명을 가져옵니다.
 1. 구성 파일 –credentials및config파일은 명령을 실행할 때 업데이트됩니다aws configure. 파일Linux 또는 macOS의 경우 또는Windows의 경우config있습니다~/.aws/configC:\Users\USERNAME\.aws\config
 1. Amazon EC2 인스턴스 프로필 자격 증명 – IAM 역할을 각 Amazon Elastic Compute Cloud(Amazon EC2) 인스턴스와 연결할 수 있습니다. 그러면 해당 역할에 대한 임시 자격 증명을 인스턴스에서 실행 중인 코드에 사용할 수 있습니다. 자격 증명은 Amazon EC2 메타데이터 서비스를 통해 전달됩니다. 자세한 내용은 Linux 인스턴스용 Amazon EC2 사용 설명서 의 Amazon EC2에 대한 IAM 역할 및 IAM 사용 설명서 의 인스턴스 프로파일 사용을 참조하십시오 .
 1. 컨테이너 자격 증명 – IAM 역할을 각 Amazon Elastic Container Service(Amazon ECS) 작업 정의와 연결할 수 있습니다. 그러면 해당 역할에 대한 임시 자격 증명을 해당 작업의 컨테이너에서 사용할 수 있습니다. 자세한 내용은 Amazon Elastic Container Service Developer Guide 의 작업에 대한 IAM 역할을 참조하십시오 .

Q. AWS CLI 및 AWS SDK는 AWS 액세스 키를 사용하여 API 요청에 서명합니다. 사용자 지정 코드를 작성하는 경우 .............................을 사용하여 AWS API 요청에 서명해야 합니다
A.  서명버전 4 (Sig V4)

Q. S3에 업로드하려고 하는 25GB 파일이 있지만 오류가 발생합니다. 이것의 가능한 원인은 무엇일까요?
A.5GB보다 큰 파일을 업로드 할 때 멀티파트 업로드를 사용한다.

Q. dev라는 새 S3 버킷을 생성하는 동안 오류가 발생합니다. 이전에 생성된 S3 버킷이 없는 새 AWS 계정을 사용하고 있습니다. 이것의 가능한 원인은 무엇일까요?
A. S3 버킷 이름은 전역적으로 고유해야하며, dev는 이미 사용중일것이다?
NOT A. S3버킷을 생성할 수 있는 IAM권한이 없다.

Q. 이미 많은 파일이 포함된 S3 버킷에서 버전 관리를 활성화했습니다. 기존 파일의 버전은 무엇일까요?
A. null

Q. IAM 사용자가 S3 버킷에서 파일을 읽기/쓰기할 수 있도록 S3 버킷 정책을 업데이트했는데, 어떤 사용자가 PutObject API 호출을 수행할 수 없다고 불평합니다. 원인은 무엇일까요?
A.IAM사용자는 첨부된 IAM정책에 명시적인 DENY가 있어야 함

Q. S3 버킷의 모든 콘텐츠를 다른 AWS 리전에서 사용할 수 있도록 하면 팀은 가능한 한 가장 적은 지연 시간과 비용으로 데이터 분석을 수행할 수 있습니다. 어떤 S3 기능을 사용해야 하나요?
A. S3 복제

Q. S3 버킷이 세 개 있습니다. A는 소스 버킷이고, B와 C는 서로 다른 AWS 리전에 있는 대상 버킷입니다. 버킷 A에서 버킷 B와 C로 객체를 복제하려면 어떻게 해야 하나요?
A. 버킷 A에서 버킷 B로 복제한 다음, 버킷A에서 버킷C로 복제하도록 구성


### S3 성능 ###
S3 기준 성능에 대해 이야기해야 합니다
기본적으로 Amazon S3는 자동으로 많은 수의 요청에 따라 자동으로 확장되며 S3에서 첫 바이트를 받는 데 100-200밀리초의 매우 낮은 지연 시간이 있습니다.
1초에 받을 수 있는 요청 수를 기준으로 하면 1초에 3,500개의 PUT/COPY/POST/DELETE를 받을 수 있으며, 버킷의 접두사마다 5,500개의 요청을 받을 수 있습니다.
이는 웹 사이트에서 얻을 수 있는 정보이며 명확하지 않기 때문에 초당 접두사당이 무슨 뜻인지 설명해 보겠습니다. 하지만 전반적인 의미는 성능이 매우 높다는 뜻이며, 버킷의 접두사 개수에는 제한이 없습니다.
4개의 객체 이름 파일의 예시를 들어 해당 객체의 접두사를 분석해 보겠습니다
첫 번째는 bucket/folder1/sub1/file에 있습니다
이 경우 접두사는 bucket과 file 사이에 있는 모든 것입니다
따라서 이 경우 /folder1/sub1입니다
다시 말해 이 파일의 경우 이 접두사에서 초당 3,500개의 put과 5,500개의 get 작업을 수행할 수 있습니다.
다른 folder1/sub2/file의 경우 접두사는 bucket과 file 사이에 있는 무엇이든 될 수 있기 때문에 folder1/sub2입니다.
따라서 이 한 개의 접두사에서도 초당 3,500개의 put과 5,500개의 get 작업을 수행할 수 있습니다
1과 2의 경우에도 접두사가 다르기 때문에 이제 접두사가 무엇인지 쉽게 이해하실 수 있을 것입니다
따라서 버킷의 접두사당 초당 3,500개의 put과 5,500개의 get 작업이 무엇인지 쉽게 이해하실 수 있을 것입니다. 즉 위의 접두사 4개에서 모두 고르게 읽으면 초당 HEAD와 GET에서 22,000개의 요청을 달성할 수 있습니다
S3 성능과 어떻게 최적화할 수 있는지에 대해 얘기해 보겠습니다

* 첫 번째는 멀티파트 업로드로 100MB보다 큰 파일의 경우 멀티파트 업로드가 권장되며
5GB보다 큰 파일의 경우 필수입니다
멀티파트 업로드는 업로드를 병렬화하여 대역폭을 극대화할 수 있도록 전송 속도를 높이는 데
도움을 줍니다.

이해를 돕기 위해 다이어그램을 사용해 보겠습니다
큰 파일이 있고 이 파일을 Amazon S3에 업로드하려면 여러 부분으로 나눌 것입니다
이러면 각 부분 파일의 덩어리가 작아져 Amazon S3에 병렬로 업로드됩니다

Amazon S3는 스마트하기 때문에 모든 부분이 업로드되면 다시 하나의 큰 파일로
합칩니다
다음은 S3 Transfer Acceleration으로 업로드와 다운로드에 사용되며
AWS 엣지 위치로 파일을 전송하고 엣지 위치에서는 대상 지역의 S3 버킷으로 데이터를 전달하여
전송 속도를 높입니다
엣지 위치는 단순한 지역이 아니며 오늘날 엣지 위치는 200개를 넘고 계속 증가하고 있습니다
그래프를 통해 보여드리겠습니다
Transfer Acceleration은 멀티파트 업로드와 호환됩니다

미국에 파일이 있으며 오스트레일리아에 있는 S3 버킷에 업로드하려 합니다
미국의 엣지 위치를 통해 해당 파일을 업로드하게 되면
퍼블릭 인터넷을 사용하여 매우 빠르게 업로드됩니다.
그리고 해당 엣지 위치에서 오스트레일리아의 Amazon S3 버킷으로는 고속 프라이빗 AWS
네트워크를 통해 전송됩니다. 이를 Transfer Acceleration이라고 하는데 이는 통과하는 퍼블릭 인터넷의 양을 최소화하고 통과하는 프라이빗 네트워크의 양을 극대화하기 때문입니다.
Transfer Acceleration은 전송 속도를 높이는 가장 좋은 방법입니다.
파일을 받는 경우는 어떨까요?
가장 효율적인 방법으로 파일을 읽으려면 어떻게 해야 할까요?
S3 Byte-Range Fetch라는 기능이 있는데 파일의 특정 바이트 범위를 받아 get 작업을
병렬화하는 것입니다.

특정 바이트 범위를 가져오지 못한 경우에도 더 작은 바이트 범위로 다시 시도할 수 있으며 장애가 있는 경우에도 복원력이 개선됩니다. 따라서 이번에는 다운로드 속도를 높이는 데 사용할 수 있습니다. S3에 아주 큰 파일이 있다고 가정해 보겠습니다.
파일에서 첫 작은 바이트인 첫 번째 부분을 요청하고 다음으로 두 번째 부분, n번째 부분을
요청하고자 한다고 하겠습니다.
이 모든 일부분을 특정 Byte-Range Fetch로 요청하는데, 파일의 특정 범위만 요청하기 때문에
이를 바이트 범위라고 하며, 이 모든 요청은 병렬로 이루어질 수 있고
요점은 get 작업을 병렬화하여 다운로드 속도를 높일 수 있다는 점입니다.

두 번째 사용 사례는 파일의 일부분만 검색하는 것입니다
예를 들어 S3에 있는 파일의 첫 50바이트가 헤더이며 파일에 대한 정보를 제공하고 헤더를 발행하여 예를 들어 헤더의 첫 50바이트를 사용하여 바이트 범위를 요청하면 정보를 매우 빠르게
얻을 수 있습니다.
지금까지 S3 성능에 대해 소개했습니다
업로드 및 다운로드 속도를 높이는 방법을 살펴보았습니다

Q1. S3 버킷에 객체가 업로드될 때 알림을 받으려면 어떻게 해야 하나요?
A. S3 이벤트 알림

Q. S3 버전 관리가 활성화된 S3 버킷이 있습니다. 이 S3 버킷에는 많은 객체가 있으며, 비용을 줄이기 위해 이전 객체 버전을 제거하려고 합니다. 오래된 객체 버전을 자동으로 삭제하는 가장 좋은 방법은 무엇인가요?
A. S3 생애 주기 규칙- 만료작업 


Q. 서로 다른 티어 간에 S3 객체의 전환을 자동화하려면 어떻게 해야 할까요?
A. S3생애 주기 규칙

Q. 다중 파트 업로드를 사용하여 대용량 파일을 S3 버킷에 업로드하는 데, 네트워크 문제로 인해 S3 버킷에 미업로드 부분이 많이 생겼습니다. 미업로드 부분은 사용하지 않는 부분인데요, 비용이 발생합니다. 미업로드 부분을 제거하는 가장 좋은 방법은 무엇인가요?
A. S3 생애 주기 정책을 사용하여 오래된/미업로드 부분의 삭제를 자동화함

Q. Amazon RDS PostgreSQL을 사용하여 S3에서 파일 인덱스를 구축하려고 합니다. 이 인덱스를 구축하려면, S3에서 각 객체의 처음 250바이트를 읽어야 합니다. 파일 자체의 콘텐츠에 대한 메타데이터가 들어 있는 부분이죠. S3 버킷에는 100,000개 이상의 파일이 있으며, 데이터의 양은 50TB에 달합니다. 어떻게 하면 이 인덱스를 효율적으로 구축할 수 있을까요?
A. S3 버킷을 가로지르는 애플리케이션을 만들고, 처음250바이트에 대해 Byte Range Fetch를 실행하고, 해당 정보를 RDS에 저장한다.

Q. 온프레미스에 저장된 대규모 데이터셋을 S3 버킷에 업로드하려고 합니다. 데이터셋은 10GB 파일로 나뉘어 있습니다. 대역폭은 양호하지만 인터넷 연결이 안정적이지 않습니다. 이 데이터셋을 S3에 업로드할 때, 인터넷 연결 문제없이 빠르게 할 수 있는 가장 좋은 방법은 무엇인가요?
A. S3 다중파트 업로드 및 S3 전송 가속 사용

Q. CSV 형식으로 S3에 저장된 데이터셋의 일부를 검색하려고 합니다. 컴퓨팅 및 네트워크 비용을 최소화하기 위해, 데이터는 한 달 분량만, 10개 열 중 3개 열만 검색하려고 합니다. 어떤 것을 사용해야 하나요?
A. S3 Select

### Amazon S3의 객체 암호화 ###
다음과 같은 4가지 방법 중 하나를 사용해 S3 버킷에서 객체를 암호화할 수 있습니다

1. 첫 번째는 서버 측 암호화, SSE로 여기에는 다시 다양한 방법이 있습니다. SSE-S3는 Amazon S3 관리 키를 사용하는 서버 측 암호화로 버킷과 객체에 대해
기본적으로 활성화되어 있습니다

2. SSE-KMS는 암호화 키를 관리하는 데 KMS 키를 사용하며
SSE-C는 고객 제공 키를 사용하는데 사용자가 직접 암호화 키를 제공하며

3. 클라이언트 측 암호화의 경우 클라이언트 측에서 모든 것을 암호화하고 Amazon S3에 업로드하는 방식입니다

시험에서는 상황에 따라 어떤 암호화 방법이 가장 잘 맞는지 아는 것이 중요하며 각각의 특성을 자세히 알아보겠습니다.

1. 첫 번째는 Amazon S3 SSE-S3 암호화입니다
이 경우 AWS에서 처리, 관리, 소유하는 키를 사용하여 암호화합니다. 사용자는 키에 액세스할 수 없습니다.
객체는 AWS에 의해 서버 측에서 암호화되며 암호화의 보안 유형은 AES-256입니다.
따라서 SSE-S3 메커니즘을 사용하여 객체를 암호화하도록 Amazon S3에 요청하려면 헤더를 "x-amz-server-side-encryption": "AES256"으로 설정해야 합니다.
SSE-S3는 새 버킷과 새 객체에 기본적으로 활성화됩니다
어떻게 작동할까요?
Amazon S3와 사용자가 있으며 사용자, 즉 여러분이 올바른 헤더가 있는 파일을 업로드하면 Amazon S3의 객체가 되고 Amazon S3는 S3 소유 키와 연결합니다

SSE-S3 메커니즘을 사용하고 있기 때문에 키와 객체를 혼합하여 암호화를 수행합니다. 이는 S3 버킷에 저장됩니다.
SSE-S3는 이와 같이 작동하며 다음으로 SSE-KMS가 있습니다.

이번에는 AWS와 S3 서비스에서 소유한 키에 의존하는 대신 KMS, 키 관리 서비스를 사용해 직접 키를 관리합니다
KMS를 사용하면 키에 대한 사용자 제어권을 갖게 되어 KMS 내에서 키를 직접 만들고 CloudTrail을 사용하여 키를 감사할 수 있습니다

따라서 누군가가 KMS에서 키를 사용할 때마다 이는 AWS에서 발생하는 모든 작업을 기록하는 서비스인 CloudTrail에 기록됩니다
사용하려면 "x-amz-server-side-encryption": "aws:kms"라는 헤더를 사용해야 하며 객체는 서버 측에서 암호화됩니다
따라서 SSE의 모든 것은 서버 측에 있습니다. 어떻게 작동할까요?
이번에도 다른 헤더가 있는 객체를 업로드하며 헤더에서는 사용하고자 하는 KMS 키를 지정합니다

2. 다음으로 객체가 Amazon S3에 나타나면 이번에는 사용할 KMS 키가 AWS KMS에서 나옵니다
이 두 가지 요소가 혼합되며 암호화를 이용할 수 있게 되고 이 파일은 S3 버킷에 보관됩니다
S3 버킷에서 파일을 읽으려면 객체 자체에 액세스해야 할 뿐 아니라 이 객체를 암호화하는 데 사용한 기반 KMS 키에도 액세스해야 하기 때문에 새로운 계층의 보안이 추가됩니다
SSE-KMS에는 제약이 있는데 Amazon S3에서 파일을 업로드 및 다운로드해야 했기 때문에 KMS 키를 활용해야 하며, KMS 키에는 생성 데이터 키와 같은 자체적인 API가 있어 암호화를 해제할 경우 Decrypt API를 사용해야 합니다, 
따라서 KMS 서비스에 API를 호출해야 하며, 이러한 각각의 API 호출은 매 초 API 호출의 KMS 할당량으로 집계됩니다
서비스 할당량 콘솔 사용량이 증가할 수 있지만 지역에 따라 초당 5,000-30,000개의 요청이 발생할 수 있습니다
S3 버킷의 스루풋이 아주 높으며 모든 파일이 KMS 키를 사용하여 암호화되면 조절 사용 사례가 생길 수 있습니다
이는 시험에 문제로 나올 수 있습니다

3. 다음으로 SSE-C 유형의 암호화가 있습니다

이 경우 키는 AWS 외부에서 관리하지만 그래도 서버 측 암호화인데 키를 AWS로 전송하기 때문입니다
하지만 Amazon S3는 사용자가 제공하는 암호화 키를 저장하지 않습니다. 사용 후에는 폐기됩니다

이 경우 키를 Amazon S3로 전송하기 때문에 HTTPS를 사용해야 하며 모든 요청에서 키를 HTTPS 헤더의 일부로 전달해야 합니다
사용자는 파일과 함께 키를 업로드하지만, 사용자는 키를 AWS 외부에서 관리하고
Amazon S3는 클라이언트 제공 키와 객체를  사용해 암호화를 수행한 다음 파일을 암호화된 상태로 S3 버킷에 저장합니다
그리고 물론 해당 파일을 읽으려면 사용자는 다시 파일을 암호화하는 데 사용한 키를 제공해야 합니다

마지막으로 클라이언트 측 암호화인데, 클라이언트 측 암호화 라이브러리와 같은 클라이언트 라이브러리를 활용하면 훨씬 쉽게 구현할 수 있습니다.
클라이언트 측 암호화의 요점은 클라이언트가 Amazon S3에 데이터를 전송하기 전 직접 암호화해야 한다는 점입니다, 또한 Amazon S3에서 데이터를 검색하고 데이터의 암호화 해제는 Amazon S3 외부의 클라이언트에서 이루어집니다
따라서 클라이언트가 키와 암호화 사이클을 전적으로 관리하게 됩니다.

파일과 AWS 외부에 있는 클라이언트 키가 있습니다. 클라이언트 자체에서 암호화를 제공하고 수행하는데 현재 암호화된 파일을 갖고 있으며, 이 파일은 그 상태로 Amazon S3에 업로드를 위해 전송될 수 있습니다
객체 암호화의 모든 수준에 대해 살펴보았지만 전송에서 일어나는 암호화에 대해서도 알아보아야 합니다. 전송 중 또는 이동 중의 암호화는 SSL이나 TLS라고도 합니다
기본적으로 Amazon S3 버킷에는 두 개의 엔드포인트가 있는데 암호화되지 않은 HTTP 엔드포인트와 이동 중 암호화되는 HTTPS 엔드포인트입니다.
웹 사이트에 방문했을 때 초록색 자물쇠나 자물쇠가 보이면 이는 보통 이동 중 암호화를 사용하여 여러분과 대상 서버 간의 연결이 안전하며 완전히 암호화되었다는 뜻입니다. 따라서 Amazon S3를 사용할 때는 단연 안전한 데이터 전송을 위해 HTTPS를 사용하는 것이 좋습니다.

그리고 SSE-C 유형의 메커니즘을 사용하는 경우 HTTPS 프로토콜을 사용해야 합니다.
현실적으로는 걱정해야 할 문제가 아닌데, 대부분의 클라이언트는 기본적으로 HTTPS 엔드포인트를 사용하기 때문입니다.

Q. 
A.

Q. 
A.

Q. 
A.

Q. 
A.

Q. 
A.

Q. 
A.

