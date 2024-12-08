import os
from pathlib import Path
from constructs import Construct
from aws_cdk import App, Stack, Environment, Duration, CfnOutput
from aws_cdk.aws_lambda import (
    DockerImageFunction,
    DockerImageCode,
    Architecture,
    FunctionUrlAuthType,
)
from aws_cdk.aws_ecr_assets import Platform

my_environment = Environment(
    account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"]
)


class GradioLambdaFunction(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create Lambda function
        lambda_fn = DockerImageFunction(
            self,
            "DeepLearningClassifier",
            code=DockerImageCode.from_image_asset(
                str(Path.cwd()), file="Dockerfile", platform=Platform.LINUX_AMD64,
            ),
            architecture=Architecture.X86_64,
            memory_size=3008,  # 3GB memory
            timeout=Duration.minutes(5),
        )

        # Add HTTPS URL
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)

        CfnOutput(self, "functionUrl", value=fn_url.url)


app = App()
gradio_lambda = GradioLambdaFunction(app, "GradioLambdaApp", env=my_environment)
app.synth()
