# Runtime Strategies And GraphRAG Chain

## Agent Runtime Strategies

The Agent supports five runtime strategies:

| Strategy | When used | Purpose |
| --- | --- | --- |
| Direct Policy | simple page and low-risk task | lowest latency action generation |
| RAG Enhanced | normal task with useful operation memory | retrieve similar trajectories and rules before deciding |
| GraphRAG | complex form, checkout, login, or multi-step page | retrieve an executable state-action path from the operation graph |
| Reflection Retry | repeated or invalid action detected | repair the plan and avoid action loops |
| RL Rollout | training/evaluation mode | sample actions, assign rewards, and optimize with PPO/GRPO |

The route is implemented in `mm_webagent/workflow/strategy_router.py`.

## GraphRAG Chain

GraphRAG replaces the previous Hybrid RAG as the main retrieval route:

```text
instruction + page_state + history
  -> graph query construction
  -> retrieve task/page/action/rule/failure nodes
  -> traverse page_state -> action -> outcome paths
  -> rank paths by page/action/rule overlap and failure penalties
  -> inject suggested action path into Agent prompt
```

Graph nodes:

- `task`: user goal and task category
- `page`: A11y Tree/HTML/Markdown page state
- `element`: button, textbox, link, tab, checkbox, menu
- `action`: `click`, `fill`, `scroll`, `goto`, terminal actions
- `outcome`: successful next state or terminal completion
- `rule`: site boundary, safety rule, or route constraint
- `failure`: repeated action, invalid element, unsafe navigation

Graph edges:

- `starts_on`: task begins at a page state
- `contains_element`: page contains a UI element
- `selects_action`: page state chooses an action
- `leads_to`: action leads to an outcome
- `constrains`: rule constrains an action
- `fails_by`: action or page pattern causes a failure

## Why GraphRAG Fits Web Agents

Hybrid RAG retrieves relevant text. GraphRAG retrieves operation structure:

```text
current page type -> available element -> historically successful action -> next state
```

This makes it better for multi-step browser tasks where the Agent needs a path,
not just a paragraph of advice.
