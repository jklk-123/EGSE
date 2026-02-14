"""
EvoTrainer: Evolutionary Training Framework for Continual Learning

This module implements an evolutionary approach to continual learning that:
- Dynamically grows a modular network architecture
- Uses reinforcement learning (PPO) to discover optimal module compositions
- Employs CKA (Centered Kernel Alignment) similarity for module reuse decisions
- Maintains task-specific pathways through the network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import json
import random
from typing import List, Dict, Tuple, Optional

from controller.policy import PolicyNetwork
from controller.ppo_trainer import PPOTrainer
from module.expert_module import ExpertModule, InputHead, OutputHead
from graph.evo_graph import EvoGraph
from graph.graph_executor import TaskNetwork
from utils.reply import Storage
from test import ContinualEvaluator
from log.logger import StructuredLogger


class EvoTrainer:
    """
    Main trainer class for evolutionary continual learning.
    
    Attributes:
        config (dict): Configuration dictionary containing hyperparameters
        device (torch.device): Device for computation (CPU/GPU)
        graph (EvoGraph): Graph structure managing module pool and paths
        evaluator (ContinualEvaluator): Evaluation handler for all tasks
        Task (List[TaskNetwork]): List of trained task networks
    """
    
    def __init__(self, config: dict, task_id: Optional[int] = None):
        """
        Initialize the EvoTrainer.
        
        Args:
            config: Configuration dictionary with training parameters
            task_id: Optional specific task ID to train
        """
        self.config = config
        self.device = config['device']
        self.repeat = config['repeat']
        
        # Initialize network heads
        self.input_head = InputHead(config).to(self.device)
        self.output_head = OutputHead(
            config['hidden_dim'], 
            config['num_classes']
        ).to(self.device)
        
        # Initialize graph structure
        self.graph = EvoGraph(config)
        
        # Setup data streams
        benchmark = config['data']
        self.train_stream = benchmark.train_stream
        self.test_stream = benchmark.test_stream
        
        # Initialize evaluator
        self.evaluator = ContinualEvaluator(config=config)
        
        # Task tracking
        self.Task = []
        self.results = 0
        
        # Logging
        self.logger = StructuredLogger(config['log_dir'])

    def setup_data_loaders(self, task_id: int):
        """
        Setup train and test data loaders for a specific task.
        
        Args:
            task_id: ID of the task to setup loaders for
        """
        self.train_loader = torch.utils.data.DataLoader(
            self.train_stream[task_id].dataset,
            batch_size=self.config['batch_train_size'],
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_stream[task_id].dataset,
            batch_size=self.config['batch_test_size'],
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False
        )
        
        self.evaluator.add_task_test_loader(
            task_id,
            self.test_loader,
            f"Task_{task_id}"
        )
    
    def setup_combined_test_loader(self, task_id: int):
        """
        Create a combined test loader for all tasks up to task_id.
        
        Args:
            task_id: Current task ID (inclusive)
        """
        from torch.utils.data import ConcatDataset
        
        combined_datasets = [
            self.test_stream[task].dataset 
            for task in range(task_id + 1)
        ]
        combined_test_dataset = ConcatDataset(combined_datasets)
        
        self.combined_test_loader = torch.utils.data.DataLoader(
            combined_test_dataset,
            batch_size=self.config['batch_test_size'],
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False
        )

    def initialize_module_pool(self) -> List[List[ExpertModule]]:
        """
        Initialize the module pool with empty expert modules.
        
        Returns:
            List of module lists for initial pathway
        """
        self.graph.history_paths[0] = [[0], [1], [2], [3]]
        pool = []
        
        for i in range(self.config['max_steps']):
            new_module = ExpertModule(self.config['hidden_dim']).to(self.device)
            self.graph.module_pool.append(new_module)
            pool.append([new_module])
        
        return pool

    def train_task_network(
        self, 
        task_id: int, 
        iteration: int = 0,
        modules: Optional[List] = None,
        prob: Optional[List] = None,
        num_epochs: int = 1000
    ):
        """
        Train a task network with specified modules and routing probabilities.
        
        Args:
            task_id: ID of current task
            iteration: Training iteration number
            modules: List of module lists for each step
            prob: Routing probabilities for modules
            num_epochs: Number of training epochs
        """
        # Create task network
        task_net = TaskNetwork(self.input_head, modules, self.output_head)
        optimizer = torch.optim.Adam(task_net.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs, 
            eta_min=5e-5
        )
        
        replay_idx = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            
            for x, y, _ in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                output = task_net.forward(x, prob)
                loss = nn.CrossEntropyLoss()(output, y)
                
                # Add regularization for previous tasks
                if iteration != -1 and task_id > 0:
                    with torch.no_grad():
                        prev_output = self.history_network[replay_idx].forward(x, prob)
                    
                    # Knowledge distillation loss
                    distill_loss = task_id * (
                        prev_output - self.Task[replay_idx].forward(x, prob)
                    ).abs().mean()
                    loss = loss + distill_loss
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Update replay index
            if iteration == -1:
                replay_idx = 0
            else:
                replay_idx = epoch % task_id if task_id > 0 else 0
            
            print(f"Iteration {iteration}, Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # Periodic evaluation
            if epoch % self.config['eval_steps'] == 0:
                print("-" * 80)
                self._evaluate_progress(task_id, epoch, task_net, prob)
                print("-" * 80)
            
            scheduler.step()
        
        # Save final state
        if iteration == -1:
            self.graph.history_probs[f'{task_id}'] = prob
            self.Task.append(task_net)
        
        print(f"History paths: {self.graph.history_paths}")

    def _evaluate_progress(
        self, 
        task_id: int, 
        epoch: int, 
        task_net: TaskNetwork, 
        prob: List
    ):
        """Helper method to evaluate and save progress during training."""
        if task_id > 0:
            history_probs = self.graph.history_probs.copy()
            history_probs[f'{task_id}'] = prob
            
            results, h_results = self.evaluator.evaluate_all_tasks(
                self.Task + [task_net], 
                history_probs, 
                task_id, 
                epoch
            )
            
            if np.mean(results) > self.results:
                self.results = np.mean(results)
                self._save_checkpoint(task_id, epoch)
            
            print(f"Eval - Results: {results}, Best: {self.results:.4f}, History: {h_results}")
        else:
            results = self.evaluator._evaluate_single_task(task_net, prob, 0)
            if results > self.results:
                self.results = results
                self._save_checkpoint(task_id, epoch)
            
            print(f"Eval - Results: {results:.4f}, Best: {self.results:.4f}")

    def _save_checkpoint(self, task_id: int, epoch: int):
        """Save model checkpoint and graph structure."""
        # Save model
        torch.save(
            [self.input_head, self.output_head] + self.graph.module_pool,
            self.config['save_dir'] + f'model_Task_{task_id}.pth'
        )
        
        # Save graph structure
        self.logger.save_graph_structure(
            task_id,
            epoch=epoch,
            paths=self.graph.history_paths,
            probs=self.graph.history_probs
        )

    def sample_trajectory(
        self, 
        candidate_modules: List[int], 
        use_max_action: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample trajectories through the module graph using the policy network.
        
        Args:
            candidate_modules: List of candidate module indices
            use_max_action: If True, use argmax instead of sampling
            
        Returns:
            Tuple of (paths, probabilities)
        """
        # Initialize state variables
        sum_modules = torch.zeros(self.repeat, 1).to(self.device)
        step_counter = torch.zeros(self.repeat, 1).to(self.device)
        paths = torch.zeros(self.repeat, self.config['steps'], self.config['act_dim']).to(self.device)
        probs = torch.zeros(self.repeat, self.config['steps'], self.config['act_dim']).to(self.device)
        
        # Expand candidate modules for batch
        modules_tensor = torch.tensor(
            candidate_modules, 
            device=self.device
        ).unsqueeze(0).repeat(self.repeat, 1)
        
        for t in range(self.config['steps']):
            # Construct state
            state = torch.cat([
                step_counter,
                sum_modules,
                modules_tensor,
                paths.reshape(paths.size(0), -1),
                probs.reshape(probs.size(0), -1)
            ], dim=-1)
            
            # Get action from policy
            with torch.no_grad():
                action_logits, param_mean, param_std, value = self.policy(state)
            
            # Sample or select action
            if use_max_action:
                action_dist = torch.distributions.Independent(
                    torch.distributions.Categorical(logits=action_logits),
                    reinterpreted_batch_ndims=1
                )
                operation = action_logits.max(-1)[1]
                param = torch.gather(param_mean, -1, operation.unsqueeze(-1)).squeeze(-1)
                param_dist = torch.distributions.Normal(
                    param, 
                    param_std * (1 - self.progress)
                )
            else:
                action_dist = torch.distributions.Independent(
                    torch.distributions.Categorical(logits=action_logits),
                    reinterpreted_batch_ndims=1
                )
                operation = action_dist.sample()
                selected_param = torch.gather(
                    param_mean, -1, operation.unsqueeze(-1)
                ).squeeze(-1)
                param_dist = torch.distributions.Normal(
                    selected_param, 
                    param_std * (1 - self.progress)
                )
                param = param_dist.sample()
            
            # Store trajectory
            paths[:, t, :] = operation
            probs[:, t, :] = param
            
            # Update state
            terminal = 1.0 if t == self.config['steps'] - 1 else 0.0
            terminals = torch.full((self.repeat, 1), terminal).to(self.device)
            log_prob = action_dist.log_prob(operation) + param_dist.log_prob(param).sum(-1)
            
            step_counter = step_counter + 1
            sum_modules += (operation > 0).sum(-1).unsqueeze(-1)
            
            # Store in replay buffer
            self.storage.add({
                's': state,
                'm': terminals,
                'a': operation,
                'p': param,
                'log_probs': log_prob,
                'v': value,
                'num': (operation > 0).sum(-1).unsqueeze(-1)
            })
        
        return paths, probs

    def build_path_modules(
        self, 
        mode: str = 'pre_module', 
        path: Optional[List] = None
    ) -> Tuple[List, List]:
        """
        Build module lists based on the specified mode and path.
        
        Args:
            mode: 'super_net' for random selection, 'pre_module' for path-based
            path: Specific path through the module graph
            
        Returns:
            Tuple of (module lists, frozen module indices)
        """
        path_modules = []
        frozen_indices = []
        used_indices = set()
        
        for i in range(self.config['max_steps']):
            if mode == 'super_net':
                # Random module selection strategy
                use_new = np.random.random() <= 0.5
                if use_new and i < len(self.new_modules):
                    path_modules.append(self.new_modules[i])
                else:
                    module = self._get_available_module(used_indices)
                    path_modules.append(module)
                    frozen_indices.append(len(path_modules) - 1)
            
            elif mode == 'pre_module':
                # Path-based module selection
                step_modules = []
                step_frozen = []
                
                for j in range(self.config['act_dim']):
                    if j == self.config['act_dim'] - 1:
                        if path[i][j] > 0:
                            step_modules.append(self.new_modules[i])
                    else:
                        if path[i][j] > 0 and path[i][j] != 10:
                            step_modules.append(
                                self.graph.module_pool[int(path[i][j])]
                            )
                            step_frozen.append(len(step_modules) - 1)
                        elif path[i][j] == 10:
                            step_modules.append(self.graph.module_pool[0])
                            step_frozen.append(len(step_modules) - 1)
                
                path_modules.append(step_modules)
                frozen_indices.append(step_frozen)
        
        return path_modules, frozen_indices

    def _get_available_module(self, used_indices: set) -> ExpertModule:
        """Get an available module from the pool."""
        available = [
            i for i in range(len(self.graph.module_pool)) 
            if i not in used_indices
        ]
        
        if available:
            idx = np.random.choice(available)
            used_indices.add(idx)
            return self.graph.module_pool[idx]
        
        return self.graph.module_pool[0]

    @torch.no_grad()
    def compute_cka_similarity(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> float:
        """
        Compute Centered Kernel Alignment (CKA) similarity between two representations.
        
        Args:
            x: First representation tensor
            y: Second representation tensor
            
        Returns:
            CKA similarity score
        """
        # Flatten representations
        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)
        
        # Center the representations
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices
        k_x = x @ x.T
        k_y = y @ y.T
        
        # Compute CKA
        numerator = (k_x * k_y).sum() ** 2
        denominator = (k_x * k_x).sum() * (k_y * k_y).sum()
        
        if denominator == 0:
            return 0.0
        
        return (numerator / denominator).item()

    def evolutionary_module_search(self, task_id: int, iteration: int):
        """
        Search for optimal module composition using evolutionary strategy.
        
        Args:
            task_id: Current task ID
            iteration: Current iteration number
        """
        # Initial training
        if iteration == 0:
            self.train_task_network(
                task_id, 
                iteration, 
                [[self.new_module]], 
                prob=[[1]], 
                num_epochs=self.config['module_epochs']
            )
        else:
            self.train_task_network(
                task_id,
                iteration,
                self.path_modules + [[self.new_module]],
                prob=self.probs + [[1]],
                num_epochs=self.config['module_epochs']
            )
        
        # Compute similarity with existing modules
        similarities = []
        for x, _, _ in self.train_loader:
            x = x.to(self.device)
            
            with torch.no_grad():
                embedding = self.input_head.forward(x)
                new_output = self.new_module.forward(embedding)
            
            for module in self.graph.module_pool:
                with torch.no_grad():
                    existing_output = module.forward(embedding)
                    sim = self.compute_cka_similarity(
                        new_output[:32], 
                        existing_output[:32]
                    )
                    similarities.append(sim)
            break
        
        print(f"Similarities: {similarities}")
        
        # Decision: Add new module or reuse existing
        if max(similarities) < 0.98:
            # Add new module to pool
            self._add_new_module()
        else:
            # Search for optimal composition
            self._search_module_composition(task_id, iteration, similarities)

    def _add_new_module(self):
        """Add the current new module to the pool."""
        self.path_modules += [[self.new_module]]
        self.probs += [[1]]
        self.history_path.append([len(self.graph.module_pool)])
        self.graph.module_pool.append(self.new_module)

    def _search_module_composition(
        self, 
        task_id: int, 
        iteration: int, 
        similarities: List[float]
    ):
        """Search for optimal module composition using RL."""
        # Select top candidates based on similarity
        top_candidates = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True
        )[:4]
        
        print(f"Top candidates: {top_candidates}, Pool size: {len(self.graph.module_pool)}")
        
        # RL-based composition search
        for rl_epoch in range(self.config['rl_epochs']):
            self.progress = rl_epoch / self.config['rl_epochs']
            self.storage = Storage(self.config['steps'])
            
            # Sample trajectories
            self.sample_trajectory(top_candidates)
            
            # Compute rewards
            rewards = self._compute_trajectory_rewards(top_candidates)
            
            # Add rewards to storage
            for step in range(rewards.size(-1)):
                self.storage.add({'r': rewards[:, step:step+1]})
            
            print(f"Step rewards: {self.storage.r[0].mean():.4f}, "
                  f"{self.storage.r[1].mean():.4f}")
            print(f"{'*' * 80} RL Epoch {rl_epoch}")
            
            # PPO update
            self.storage.add({'v': torch.zeros([1]).to(self.device)})
            policy_loss, value_loss = self.ppo.update_policy(self.storage, self.progress)
            
            # Log metrics
            metrics = {
                'total_reward': (self.storage.r[0] + self.storage.r[1]).mean().item(),
                'policy_loss': policy_loss,
                'value_loss': value_loss,
            }
            print(f"Metrics: {metrics}")
            
            self.logger.log_metrics(
                task_id=task_id,
                epoch=rl_epoch + self.config['rl_epochs'] * iteration,
                progress=self.progress,
                metrics=metrics
            )
        
        # Finalize composition based on performance
        if self.storage.r[0].mean().item() > 0.99:
            self._finalize_composition(top_candidates)
        else:
            self._add_new_module()

    def _compute_trajectory_rewards(
        self, 
        candidate_indices: List[int]
    ) -> torch.Tensor:
        """Compute rewards for sampled trajectories."""
        # Get sample batch
        for x, _, _ in self.train_loader:
            x = x.to(self.device)
            break
        
        rewards = torch.zeros(self.repeat, 2).to(self.device)
        
        with torch.no_grad():
            embedding = self.input_head.forward(x)
            target_output = self.new_module.forward(embedding)
        
        for traj_idx in range(self.repeat):
            y = embedding
            
            for step in range(self.config['steps']):
                # Compute weighted combination of modules
                step_output = torch.zeros_like(y)
                
                for action_idx in range(4):
                    with torch.no_grad():
                        module_output = self.graph.module_pool[
                            candidate_indices[action_idx]
                        ].forward(y)
                        
                        weight = (
                            self.storage.a[step][traj_idx, action_idx] *
                            self.storage.p[step][traj_idx, action_idx]
                        )
                        step_output += weight * module_output
                
                # Compute step reward
                with torch.no_grad():
                    similarity = self.compute_cka_similarity(
                        target_output[:32], 
                        step_output[:32]
                    )
                    complexity_penalty = 0.1 * F.relu(
                        self.storage.num[0][traj_idx] - 3
                    )
                    rewards[traj_idx, step] = similarity - complexity_penalty
                
                y = step_output
        
        return rewards

    def _finalize_composition(self, candidate_indices: List[int]):
        """Extract and save the best module composition."""
        paths, probs = self.sample_trajectory(candidate_indices, use_max_action=True)
        print(f"Final composition - Paths: {paths[0]}, Probs: {probs[0]}")
        
        for step in range(probs[0].size(0)):
            step_modules = []
            step_probs = []
            step_indices = []
            
            for action_idx in range(probs[0].size(1)):
                if paths[0, step, action_idx] > 0:
                    step_modules.append(
                        self.graph.module_pool[candidate_indices[action_idx]]
                    )
                    step_probs.append(probs[0, step, action_idx].item())
                    step_indices.append(candidate_indices[action_idx])
            
            if len(step_modules) > 0:
                self.path_modules.append(step_modules)
                self.probs.append(step_probs)
                self.history_path.append(step_indices)

    def train_task(self, task_id: int):
        """
        Complete training procedure for a single task.
        
        Args:
            task_id: ID of the task to train
        """
        # Setup
        self.history_network = [
            copy.deepcopy(self.Task[i]) 
            for i in range(task_id)
        ]
        self.setup_combined_test_loader(task_id)
        
        self.results = 0
        self.probs = []
        self.path_modules = []
        self.history_path = []
        
        # Evolutionary search for each module position
        for step in range(self.config['max_steps']):
            self.policy = PolicyNetwork().to(self.device)
            self.ppo = PPOTrainer(self.policy, self.config)
            self.new_module = ExpertModule(self.config['hidden_dim']).to(self.device)
            
            self.evolutionary_module_search(task_id, step)
        
        # Save discovered architecture
        self.save_architecture(
            task_id=task_id,
            path=self.history_path,
            prob=self.probs
        )
        
        # Final training with discovered architecture
        self.train_task_network(
            task_id,
            self.config['max_steps'],
            self.path_modules,
            prob=self.probs,
            num_epochs=500
        )
        
        # Create and store final task network
        task_network = TaskNetwork(
            self.input_head,
            self.path_modules,
            self.output_head,
        ).to(self.device)
        self.Task.append(task_network)

    def save_architecture(
        self, 
        task_id: int, 
        path: List, 
        prob: List, 
        epoch: int = 0
    ):
        """
        Save the discovered architecture for a task.
        
        Args:
            task_id: Task identifier
            path: Module pathway
            prob: Routing probabilities
            epoch: Current epoch
        """
        self.graph.history_paths[f'{task_id}'] = path
        self.graph.history_probs[f'{task_id}'] = prob
        
        self.logger.save_graph_structure(
            task_id,
            epoch=epoch,
            paths=self.graph.history_paths,
            probs=self.graph.history_probs
        )
        
        print(f"Saved architecture - Paths: {self.graph.history_paths}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, list):
            self.input_head = checkpoint[0].to(self.device)
            self.output_head = checkpoint[1].to(self.device)
            self.graph.module_pool = checkpoint[2:]
        
        print(f"Loaded checkpoint from {checkpoint_path}")

    def evaluate_task_assignment(
        self, 
        task_id: int, 
        task_networks: List[TaskNetwork], 
        history_probs: Dict
    ):
        """
        Evaluate using task assignment based on confidence.
        
        Args:
            task_id: Current task ID
            task_networks: List of task networks
            history_probs: Routing probabilities for each task
        """
        correct = 0
        total = 0
        
        for x, y, *extra in self.combined_test_loader:
            # Handle different label formats
            if isinstance(y, (list, tuple)) and len(y) > 0:
                y = y[0]
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Get predictions from all task networks
            all_logits = []
            for i in range(task_id + 1):
                with torch.no_grad():
                    outputs = task_networks[i](x, history_probs[str(i)])
                all_logits.append(outputs)
            
            all_logits = torch.stack(all_logits, dim=0)
            
            # Select predictions based on max confidence
            confidence_scores = F.softmax(all_logits, dim=-1)
            max_probs, pred_classes = confidence_scores.max(dim=-1)
            
            best_task_idx = max_probs.argmax(dim=0)
            predictions = pred_classes.gather(
                0, best_task_idx.unsqueeze(0)
            ).squeeze(0)
            
            total += y.size(0)
            correct += predictions.eq(y).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        print(f"Task assignment accuracy: {accuracy:.2f}%")
        
        return accuracy
