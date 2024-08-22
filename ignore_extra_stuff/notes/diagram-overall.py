from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS, Dynamodb
from diagrams.aws.network import ELB, Route53, CloudFront, VPC
from diagrams.aws.storage import S3
from diagrams.aws.security import KMS
from diagrams.onprem.client import Users, Client
from diagrams.onprem.network import Nginx
from diagrams.onprem.compute import Server
from diagrams.onprem.container import Docker
from diagrams.onprem.ci import Jenkins
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.onprem.logging import FluentBit
from diagrams.onprem.queue import Kafka
from diagrams.onprem.inmemory import Redis
from diagrams.generic.database import SQL
from diagrams.generic.storage import Storage
from diagrams.saas.chat import Slack

with Diagram("Wonderschool-like Architecture", show=False):

    users = Users("Users")

    with Cluster("CDN and DNS"):
        cdn = CloudFront("CDN")
        dns = Route53("DNS")

    with Cluster("Web Servers"):
        lb = ELB("Load Balancer")
        webserver1 = EC2("Web Server 1")
        webserver2 = EC2("Web Server 2")

    with Cluster("Backend Services"):
        backend = [
            Server("User Service"),
            Server("Program Service"),
            Server("Booking Service"),
            Server("Payment Service"),
            Server("Review Service"),
            Server("Notification Service"),
            Server("Real-time Chat Service")
        ]

    with Cluster("Databases"):
        rds = RDS("PostgreSQL")
        nosql = Dynamodb("MongoDB")
        cache = Redis("Redis")

    with Cluster("Search"):
        elasticsearch = Server("Elasticsearch")

    with Cluster("Payment Processing"):
        stripe = Server("Stripe API")

    with Cluster("Monitoring & Logging"):
        prometheus = Prometheus("Prometheus")
        grafana = Grafana("Grafana")
        fluentbit = FluentBit("FluentBit")

    with Cluster("CI/CD Pipeline"):
        ci_cd = Jenkins("Jenkins")

    with Cluster("Infrastructure"):
        vpc = VPC("VPC")
        storage = S3("File Storage")
        kms = KMS("Key Management Service")

    users >> dns >> cdn >> lb >> [webserver1, webserver2]
    webserver1 >> backend
    webserver2 >> backend

    backend >> Edge(color="darkblue") >> rds
    backend >> Edge(color="darkred") >> nosql
    backend >> Edge(color="purple") >> cache
    backend >> Edge(color="orange") >> elasticsearch
    backend >> Edge(color="green") >> stripe

    rds >> Edge(color="black") >> storage
    nosql >> Edge(color="black") >> storage

    prometheus >> grafana
    backend >> prometheus
    backend >> fluentbit
    fluentbit >> grafana

    ci_cd >> backend
    ci_cd >> vpc

    kms >> storage
    kms >> rds
    kms >> nosql

