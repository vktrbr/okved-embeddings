import dgl
import torch


class NegativeSamplerRel:

    def __init__(self, k=1):
        self.k = k

    def __call__(self, graph: dgl.DGLGraph, device: str = 'cpu') -> dgl.DGLGraph:
        """
        Создает граф на основе исходного. Шаг 1 - забирает исходные данные. Шаг 2 - создаем массив [[u, r, v] * n],
        pos/neg из исходного Шаг 3 - neg samples повторяем k раз. Шаг 4 - случайно заменяем source-target пары в
        массиве neg samples

        Args:
            graph: dgl.DGLGraph
                Граф, из которого нужно возвращать негативные примеры
            device: str
                Название устройства на котором вычисляем ('cpu', 'cuda', ..)

        Returns:
            Граф с негативными примерами
        """
        source, target, num_relation = graph.edges(form='all')  # Шаг 1
        edge_types = graph.edata['type'][num_relation]

        # pos_samples: [u, r, v]
        pos_samples = torch.column_stack([source, edge_types, target]).to(device)  # Шаг 2
        neg_batch_size = len(pos_samples) * int(self.k)
        neg_samples = torch.tile(pos_samples, (int(self.k), 1)).to(device)  # Шаг 3

        values = torch.randint(graph.num_nodes(), size=(neg_batch_size,)).to(device)  # Шаг 4
        choices = torch.rand(size=(neg_batch_size,)).to(device)

        subj = (choices > torch.FloatTensor([0.5]).to(device)).to(device)
        obj = (choices <= torch.FloatTensor([0.5]).to(device)).to(device)

        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]

        neg_graph = dgl.graph((neg_samples[:, 0], neg_samples[:, 2]), num_nodes=graph.num_nodes())
        neg_graph.edata['type'] = neg_samples[:, 1]

        return neg_graph
