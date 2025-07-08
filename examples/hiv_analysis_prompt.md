# Natural Multi-Tool Analysis Question

I'm working on a public health research project and need to understand HIV patterns in India. Can you help me analyze the HIV dataset (resource ID: 9a362ec2-2cfc-4e08-8c74-7926b2159a69) to identify which states have the most concerning HIV trends and what demographic groups are most affected? I'm particularly interested in finding actionable insights for targeted public health interventions.

---

**Why this works:**
- Claude will naturally decide to load the dataset first to see what's available
- It will inspect the structure to understand the columns
- It will then perform multiple analyses based on what it finds
- The MCP server will provide column information between tool calls
- Claude will adapt its analysis strategy based on the actual data structure
- No scripted steps - just intelligent tool selection based on the research question
