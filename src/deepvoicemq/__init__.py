from docker import DockerClient

def main():
    client = DockerClient.from_env()
    client.containers.run(
        "redis:latest",
        name="deepvoice-redis",
        detach=True,
        ports={"6379/tcp": 6379},
        volumes={"/var/lib/deepvoice/redis": {"bind": "/data", "mode": "rw"}},
        restart_policy={"name": "always"}
    )
    print("DeepVoice Redis container started.")

if __name__ == "__main__":
    main()