import typer
import typer.rich_utils
from primaite.network import NetworkGenerator


def main(num_nodes: int, num_links: int, num_services: int):

    network = NetworkGenerator(num_nodes=3, num_services=2, num_links=1)
    network.save("src/primaite/config/_package_data/lay_down/test.yaml")


if __name__ == "__main__":
    typer.run(main)
