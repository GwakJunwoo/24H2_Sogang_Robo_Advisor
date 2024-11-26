
from typing import List, Optional, Tuple, Union, Callable
from Tree import *
from BaseOptimizer import *
from Optimizer import *
from Assumption import *

optimizer_inputs = {
    'mean_variance_optimizer': ['expected_returns', 'covariance_matrix'],
    'equal_weight_optimizer': [],
    'dynamic_risk_optimizer': ['covariance_matrix'],
    'risk_parity_optimizer' : ['covariance_matrix'],
    'goal_based_optimizer': ['expected_returns', 'covariance_matrix'],

}

class Pipeline:
    def __init__(self, steps: List[Tuple[str, Callable]], universe: Tree, assumption: AssetAssumption):
        self.steps = steps
        self.universe = universe
        self.assumption = assumption

    def run(self, price_data: pd.DataFrame) -> Dict[str, float]:
        # Calculate the assumptions (expected returns and covariance)
        expected_returns = self.assumption.calculate_expected_return(price_data)
        covariance_matrix = self.assumption.calculate_covariance(price_data)

        # 비정상적인 기대수익률(-99999)을 가지는 자산 배제
        valid_assets = expected_returns[expected_returns > -99999].index
        filtered_expected_returns = expected_returns.loc[valid_assets]
        filtered_covariance_matrix = covariance_matrix.loc[valid_assets, valid_assets]

        # Initialize an empty dictionary for allocations
        allocations = {}
        root_node = self.universe.root
        self._optimize_node(root_node, 1, allocations, filtered_expected_returns, filtered_covariance_matrix)
        return allocations

    def _optimize_node(
        self,
        node: Node,
        depth: int,
        allocations: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        parent_weight: float = 1.0
    ) -> None:
        """Recursively optimizes each node and its children using the specified optimizer at each depth."""
        if depth <= len(self.steps):
            optimizer_name, optimizer_func = self.steps[depth - 1]

            # Get the required inputs for this optimizer from the optimizer_inputs mapping
            required_inputs = optimizer_inputs.get(optimizer_func.__name__, [])
            input_args = {}

            # Extract the names of the child nodes
            child_names = [child.name for child in node.children]

            # 필터링된 자산을 고려하여 자식 노드 중 사용 가능한 자산만 선택
            valid_child_names = [name for name in child_names if name in expected_returns.index]

            if len(valid_child_names) == 0:
                # 모든 자식이 필터링되었다면 비중을 0으로 설정
                for child_node in node.children:
                    allocations[child_node.name] = 0.0
                return

            # Extract the relevant expected_returns and covariance_matrix for the valid child nodes
            if 'expected_returns' in required_inputs:
                input_args['expected_returns'] = expected_returns[valid_child_names].values
            if 'covariance_matrix' in required_inputs:
                input_args['covariance_matrix'] = covariance_matrix.loc[valid_child_names, valid_child_names].values

            # Retrieve weight bounds for valid child nodes and pass to optimizer
            valid_children = [child for child in node.children if child.name in valid_child_names]
            weight_bounds = self._get_nodes_bounds(valid_children)
            if weight_bounds:
                input_args['weight_bounds'] = weight_bounds

            # 명시적인 인자 전달
            if optimizer_func == mean_variance_optimizer:
                node_weights = optimizer_func(
                    valid_children,
                    expected_returns=input_args['expected_returns'],
                    covariance_matrix=input_args['covariance_matrix'],
                    weight_bounds=input_args.get('weight_bounds', None)
                )
            else:
                # Run optimizer for other optimizers (like risk_parity_optimizer, equal_weight_optimizer)
                node_weights = optimizer_func(valid_children, **input_args)

            # 최적화된 가중치를 할당하고, 자식 노드를 재귀적으로 최적화
            for child_node, weight in zip(valid_children, node_weights):
                allocations[child_node.name] = weight * parent_weight
                # Recursively optimize the children nodes
                self._optimize_node(
                    child_node,
                    depth + 1,
                    allocations,
                    expected_returns,
                    covariance_matrix,
                    weight * parent_weight
                )

            # 필터링되어 최적화 대상에서 제외된 자식 노드는 비중을 0으로 설정
            for child_node in node.children:
                if child_node.name not in valid_child_names:
                    allocations[child_node.name] = 0.0


    def _get_nodes_bounds(self, nodes: List[Node]) -> List[Tuple]:
        """Retrieves the weight bounds for a list of nodes."""
        return [node.params['weight_bounds'] for node in nodes]
