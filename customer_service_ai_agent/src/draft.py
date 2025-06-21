class CustomerServiceAgent:
    """Main AI Agent class with hybrid sync/async capabilities"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.tools = None
        self.agent = None
        self.support_index = None
        self.query_decomposer = None
        self.sync_tools = None      # Regular tools
        self.async_tools = None     # Async tools for parallel operations

    def setup_database(self):
        """Setup database connection"""
        print("🔌 Connecting to database...")
        if not self.db_manager.connect():
            return False

        # Test connection
        test_query = "SELECT COUNT(*) FROM orders"
        results = self.db_manager.execute_query(test_query)
        if results:
            print(f"✅ Database connected! Found {results[0][0]} orders.")
            return True
        return False

    def setup_llm(self):
        """Setup Local LLM connection with enhanced configuration"""
        if not LLAMAINDEX_AVAILABLE:
            return False

        print("🤖 Setting up Local LLM...")
        try:
            # Check for Mac GPU availability
            try:
                import torch
                if torch.backends.mps.is_available():
                    device_str = "mps"
                    print("🚀 Mac GPU (Metal) detected and will be used for embeddings")
                else:
                    device_str = "cpu"
                    print("💻 Using CPU for embeddings")
            except ImportError:
                device_str = "cpu"
                print("💻 PyTorch not available, using CPU for embeddings")

            # Local server configuration
            local_llm_url = "http://127.0.0.1:1234/v1"

            # Setup the LLM to use local server with optimized settings
            Settings.llm = OpenAILike(
                model="open_gpt4_8x7b_v0.2",
                api_base=local_llm_url,
                api_key="lm-studio",
                is_local=True,
                temperature=0.1,
                max_tokens=3000,
                timeout=45,
                max_retries=2
            )

            # Setup local embedding model for RAG with GPU acceleration
            try:
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    max_length=512,
                    device=device_str,
                    trust_remote_code=True
                )
            except Exception as embed_error:
                print(f"⚠️  GPU embedding failed, falling back to CPU: {embed_error}")
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    max_length=512
                )
                device_str = "cpu"

            print("✅ LLM setup completed with optimizations!")
            print(f"   🎯 Temperature: 0.1 (focused responses)")
            print(f"   📝 Max tokens: 3000 (detailed answers)")
            print(f"   ⏱️  Timeout: 45s per call")
            print(f"   🔄 Retries: 2 attempts")
            print(f"   🚀 Embedding device: {device_str}")
            return True

        except Exception as e:
            print(f"❌ LLM setup failed: {e}")
            print("💡 Make sure LM Studio is running on http://127.0.0.1:1234/v1")
            return False

    def get_support_files_from_user(self):
        """Get support files from user input"""
        print("\n📚 Support Document Configuration")
        print("=" * 50)
        print("Please specify the support documents you want to use.")
        print("Supported formats: .txt, .pdf, .doc, .docx, .pptx")
        print("Default files: Customer Service.txt, FAQ.txt, Return Policy.txt")
        print()

        while True:
            choice = input("Use default files? (y/n) or 'custom' for custom files: ").strip().lower()

            if choice in ['y', 'yes']:
                return ['Customer Service.txt', 'FAQ.txt', 'Return Policy.txt']
            elif choice in ['n', 'no', 'custom']:
                print("\nEnter support file names (one per line).")
                print("Press Enter on empty line when done:")
                files = []
                while True:
                    file = input("File name: ").strip()
                    if not file:
                        break
                    files.append(file)
                return files if files else ['Customer Service.txt', 'FAQ.txt', 'Return Policy.txt']
            else:
                print("❌ Invalid choice. Please enter 'y', 'n', or 'custom'.")

    def validate_support_files(self, files):
        """Validate that all support files exist"""
        print(f"\n🔍 Validating {len(files)} support file(s)...")

        missing_files = []
        valid_files = []

        supported_extensions = ['.txt', '.pdf', '.doc', '.docx', '.pptx']

        for file in files:
            # Check if file exists
            if not os.path.exists(file):
                missing_files.append(file)
                continue

            # Check if file has supported extension
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in supported_extensions:
                print(f"⚠️  Warning: {file} has unsupported format {file_ext}")
                print(f"   Supported formats: {', '.join(supported_extensions)}")
                continue

            valid_files.append(file)
            print(f"✅ Found: {file}")

        if missing_files:
            print(f"\n❌ Missing files:")
            for file in missing_files:
                print(f"   - {file}")
            print(f"\n💡 Please ensure all required files are in the current directory:")
            print(f"   Current directory: {os.getcwd()}")
            return None

        if not valid_files:
            print("❌ No valid support files found!")
            return None

        print(f"✅ All {len(valid_files)} support file(s) validated successfully!")
        return valid_files

    def setup_support_documents(self):
        """Setup vector index for customer support documents"""
        if not LLAMAINDEX_AVAILABLE:
            return False

        print("📚 Setting up support documents...")

        try:
            # Get support files from user
            requested_files = self.get_support_files_from_user()

            # Validate files exist
            valid_files = self.validate_support_files(requested_files)
            if not valid_files:
                print("❌ Required support files not found. Exiting program.")
                print("Please ensure all support documents are available and try again.")
                return False

            print(f"\n📁 Loading {len(valid_files)} support document(s):")
            for file in valid_files:
                file_size = os.path.getsize(file)
                print(f"   - {file} ({file_size:,} bytes)")

            # Setup vector index for support documents
            try:
                support_docs = SimpleDirectoryReader(input_files=valid_files).load_data()
                print(f"📄 Loaded {len(support_docs)} document(s)")

                splitter = SentenceSplitter(
                    chunk_size=1024,
                    chunk_overlap=50  # Add overlap for better context
                )
                support_nodes = splitter.get_nodes_from_documents(support_docs)
                print(f"🔧 Created {len(support_nodes)} text chunks")

                self.support_index = VectorStoreIndex(support_nodes)

                print(f"✅ Support documents indexed successfully!")
                print(f"   Documents: {len(valid_files)}")
                print(f"   Chunks: {len(support_nodes)}")
                print(f"   Ready for vector search!")

                return True

            except Exception as e:
                print(f"❌ Error processing support documents: {e}")
                print("Please check document formats and content.")
                return False

        except Exception as e:
            print(f"❌ Support documents setup failed: {e}")
            return False

    def create_tools(self):
        """Create enhanced tools for the agent with hybrid sync/async capabilities"""
        if not LLAMAINDEX_AVAILABLE:
            return False

        print("🛠️  Creating enhanced agent tools...")
        try:
            # Create regular database-connected tools
            self.sync_tools = CustomerServiceTools(self.db_manager)

            # Create async tools for parallel operations
            self.async_tools = AsyncCustomerServiceTools(self.db_manager, self.sync_tools)

            # Specialized return policy tool
            def get_order_return_policy(order_id: int) -> str:
                """Get return policy for all items in a specific order"""
                try:
                    order_id = int(order_id)

                    # Get order items
                    items = self.sync_tools.get_order_items(order_id)
                    if not items:
                        return f"Order {order_id} not found or has no items."

                    # Get return policy for each item
                    policies = []
                    for item in items:
                        days = self.sync_tools.get_item_return_days(item)
                        policies.append(f"- {item}: {days} days return policy")

                    return f"Return policy for order {order_id}:\n" + "\n".join(policies)

                except Exception as e:
                    return f"Error getting return policy for order {order_id}: {str(e)}"

            # Parallel multi-order tool for expert scenarios
            def get_multiple_orders_sync(order_ids: str) -> Dict[str, Any]:
                """Sync wrapper for async parallel order retrieval"""
                try:
                    # Parse order IDs
                    if isinstance(order_ids, str):
                        ids = [int(x.strip()) for x in order_ids.replace(',', ' ').split()]
                    else:
                        ids = [int(order_ids)]

                    # Use async method with asyncio.run for sync compatibility
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(
                            self.async_tools.get_multiple_orders_parallel(ids)
                        )
                        return {
                            'orders_found': len(results),
                            'orders': results
                        }
                    finally:
                        loop.close()

                except Exception as e:
                    return {"error": f"Failed to get multiple orders: {str(e)}"}

            # Create all function tools
            order_return_policy_tool = FunctionTool.from_defaults(
                fn=get_order_return_policy,
                name="get_order_return_policy",
                description="Get return policy for all items in a specific order. Input: order_id as integer (e.g., 1001)"
            )

            multiple_orders_parallel_tool = FunctionTool.from_defaults(
                fn=get_multiple_orders_sync,
                name="get_multiple_orders_parallel",
                description="Get details for multiple orders in parallel (faster). Input: order_ids as comma-separated string (e.g., '1007,1017,1023')"
            )

            # Basic function tools
            order_item_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.get_order_items,
                name="get_order_items",
                description="Get list of items in a specific order. Input: order_id as integer (e.g., 1001)"
            )

            delivery_date_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.get_delivery_date,
                name="get_delivery_date",
                description="Get delivery date for a specific order. Input: order_id as integer (e.g., 1001)"
            )

            return_policy_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.get_item_return_days,
                name="get_item_return_days",
                description="Get return policy days for a specific product name. Input: item as string (e.g., 'Laptop' or 'Mouse')"
            )

            order_details_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.get_order_details,
                name="get_order_details",
                description="Get comprehensive details for a specific order including customer info. Input: order_id as integer (e.g., 1001)"
            )

            search_orders_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.search_orders_by_customer,
                name="search_orders_by_customer",
                description="Search all orders for a customer by email address. Input: email as string (e.g., 'john.smith@email.com')"
            )

            # Advanced analytics tools
            comprehensive_analysis_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.analyze_customer_orders_comprehensive,
                name="analyze_customer_orders_comprehensive",
                description="Comprehensive analysis of all customer orders with return eligibility. Input: customer_email as string"
            )

            return_calculation_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.calculate_return_policy_and_deadlines,
                name="calculate_return_policy_and_deadlines",
                description="Calculate exact return deadlines and policy details for a specific order. Input: order_id as integer"
            )

            geographic_analysis_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.analyze_geographic_performance,
                name="analyze_geographic_performance",
                description="Analyze order patterns and performance by geographic location. Input: state as string (optional), city as string (optional)"
            )

            predictive_analysis_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.generate_predictive_risk_analysis,
                name="generate_predictive_risk_analysis",
                description="Generate predictive analysis for customer service risks and proactive recommendations. No input required."
            )

            status_analysis_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.analyze_orders_by_status_and_timeframe,
                name="analyze_orders_by_status_and_timeframe",
                description="Analyze orders by status within timeframe with delivery performance metrics. Input: start_date as string (optional), end_date as string (optional)"
            )

            product_performance_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.analyze_product_performance_and_issues,
                name="analyze_product_performance_and_issues",
                description="Analyze product performance, popularity, and identify potential issues. Input: product_category as string (optional)"
            )

            # Support document tool
            support_tool = QueryEngineTool.from_defaults(
                query_engine=self.support_index.as_query_engine(),
                name="support_policy_search",
                description="Search customer support policies, return policies, FAQ, and contact information from support documents"
            )

            self.tools = [
                # Basic tools (scenarios 1-3) - prioritize specialized tools
                order_return_policy_tool,  # Specialized for return policy queries
                order_item_tool,
                delivery_date_tool,
                return_policy_tool,
                order_details_tool,
                search_orders_tool,

                # Parallel processing tools for expert scenarios
                multiple_orders_parallel_tool,  # NEW: Faster for multi-order queries

                # Advanced analytics tools (scenarios 4-15)
                comprehensive_analysis_tool,
                return_calculation_tool,
                geographic_analysis_tool,
                predictive_analysis_tool,
                status_analysis_tool,
                product_performance_tool,

                # Support document search
                support_tool
            ]

            print(f"✅ Created {len(self.tools)} enhanced tools successfully!")
            print("   📊 Basic tools: 6 (including specialized return policy tool)")
            print("   ⚡ Parallel tools: 1 (for multi-order scenarios)")
            print("   🚀 Advanced analytics: 6 (comprehensive analysis)")
            print("   📚 Support search: 1 (policy and FAQ)")
            return True

        except Exception as e:
            print(f"❌ Tool creation failed: {e}")
            return False

    def create_agent(self):
        """Create the enhanced AI agent with better configuration"""
        if not LLAMAINDEX_AVAILABLE or not self.tools:
            return False

        print("🤖 Creating enhanced AI agent...")
        try:
            # Setup the Agent worker with optimized settings
            agent_worker = ReActAgentWorker.from_tools(
                self.tools,
                llm=Settings.llm,
                verbose=True,
                max_iterations=15,  # Increased from default 10
                allow_parallel_tool_calls=False  # Sequential for better reasoning
            )

            # Create agent runner with enhanced configuration
            self.agent = AgentRunner(
                agent_worker,
                memory=None,  # Fresh context for each query
                verbose=True
            )

            print("✅ Enhanced AI agent created successfully!")
            print("   🔧 Max iterations: 15 (increased for complex scenarios)")
            print("   🧠 Sequential reasoning: Enabled for better tool selection")
            print("   📝 Fresh context: Each query starts clean")
            print("   ⚡ Parallel processing: Available for expert scenarios")
            return True

        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
            return False

    def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing Customer Service AI Agent...")
        print("=" * 60)

        steps = [
            ("Database Connection", self.setup_database),
            ("LLM Setup", self.setup_llm),
            ("Support Documents", self.setup_support_documents),
            ("Agent Tools", self.create_tools),
            ("AI Agent", self.create_agent)
        ]

        for step_name, step_func in steps:
            print(f"\n📋 Step: {step_name}")
            if not step_func():
                print(f"❌ Failed at step: {step_name}")
                return False

        # Setup query decomposer
        print("\n📋 Step: Query Decomposition System")
        if self.setup_query_decomposer():
            print("✅ Query decomposition enabled!")
        else:
            print("⚠️  Query decomposition setup failed, using standard mode")

        print("\n" + "=" * 60)
        print("🎉 Customer Service AI Agent initialized successfully!")
        print("🚀 Performance optimizations active:")
        print("   ⚡ Parallel processing for multi-order scenarios")
        print("   🔧 Connection pooling for faster database queries")
        print("   💾 GPU acceleration for embeddings (if available)")
        return True

    def run_predefined_scenarios(self):
        """Run predefined test scenarios"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        scenarios = [
            # Basic scenarios (original)
            {
                "name": "Return Policy Query",
                "query": "What is the return policy for order number 1001?",
                "description": "Tests order lookup and return policy retrieval",
                "difficulty": "Basic"
            },
            {
                "name": "Multi-part Question",
                "query": "When is the delivery date and items shipped for order 1003 and how can I contact customer support?",
                "description": "Tests multiple tool usage in single query",
                "difficulty": "Basic"
            },
            {
                "name": "Invalid Order",
                "query": "What is the return policy for order number 9999?",
                "description": "Tests handling of non-existent orders",
                "difficulty": "Basic"
            },

            # Intermediate scenarios
            {
                "name": "Customer History Analysis",
                "query": "Show me all orders for customer john.smith@email.com and tell me which items can still be returned based on delivery dates",
                "description": "Tests customer search, multiple orders, date calculations, and policy application",
                "difficulty": "Intermediate"
            },
            {
                "name": "Product Category Return Policy",
                "query": "I bought a laptop, mouse, and HDMI cable in different orders. What are the return policies for each and which one expires first?",
                "description": "Tests product search across orders, policy comparison, and time analysis",
                "difficulty": "Intermediate"
            },
            {
                "name": "Order Status Investigation",
                "query": "Why hasn't order 1005 been delivered yet? Check the status and compare with similar orders from the same timeframe",
                "description": "Tests order details, status analysis, and comparative investigation",
                "difficulty": "Intermediate"
            },
            {
                "name": "Cross-Reference Analysis",
                "query": "Find all customers who ordered laptops in June 2024 and tell me their order statuses and delivery performance",
                "description": "Tests complex joins, date filtering, and batch analysis",
                "difficulty": "Intermediate"
            },

            # Advanced scenarios
            {
                "name": "Complex Return Calculation",
                "query": "For order 1022, calculate the exact return deadline for each item considering the delivery date was June 6th, and tell me what happens if I return only the mouse but keep the laptop",
                "description": "Tests date arithmetic, partial returns, policy calculations, and business logic",
                "difficulty": "Advanced"
            },
            {
                "name": "Customer Lifecycle Analysis",
                "query": "Analyze the complete order history for customers in Auburn, AL. Which products are most popular, what's the average order value, and identify any delivery issues",
                "description": "Tests geographic filtering, statistical analysis, trend identification, and problem detection",
                "difficulty": "Advanced"
            },
            {
                "name": "Escalation Scenario",
                "query": "Customer placed order 1013 on June 13th for a gaming laptop costing $1599.99 but it's still pending after a week. They're threatening to cancel. What are our options for resolution and how should we escalate this?",
                "description": "Tests crisis management, order prioritization, escalation procedures, and solution generation",
                "difficulty": "Advanced"
            },
            {
                "name": "Inventory Impact Analysis",
                "query": "If all pending orders from the last week need to be expedited due to customer complaints, which products would be affected and what's the total value at risk?",
                "description": "Tests order aggregation, financial calculations, business impact analysis, and risk assessment",
                "difficulty": "Advanced"
            },

            # Expert scenarios
            {
                "name": "Multi-Customer Dispute Resolution",
                "query": "Three customers (orders 1007, 1017, 1023) all received MacBook Pros but are reporting different issues: one has screen problems, one has battery issues, one has keyboard problems. Analyze their orders, determine warranty coverage, and recommend resolution strategy for each",
                "description": "Tests multi-order analysis, issue categorization, warranty determination, and personalized solutions",
                "difficulty": "Expert"
            },
            {
                "name": "Supply Chain Investigation",
                "query": "Several customers are complaining about late deliveries for orders placed in early June. Analyze delivery performance for orders 1001-1015, identify patterns, and determine if there's a systemic issue with specific product categories or shipping regions",
                "description": "Tests pattern recognition, performance analysis, root cause identification, and systemic problem detection",
                "difficulty": "Expert"
            },
            {
                "name": "Revenue Recovery Strategy",
                "query": "Customer wants to return their entire order 1007 ($2129.98) due to compatibility issues, but they've already used the laptop for 2 weeks. Analyze the order details, check our return policy exceptions, calculate potential restocking fees, and propose alternatives that minimize revenue loss while satisfying the customer",
                "description": "Tests policy interpretation, financial calculations, alternative solution generation, and business optimization",
                "difficulty": "Expert"
            },
            {
                "name": "Predictive Customer Service",
                "query": "Based on order patterns, delivery dates, and customer behavior, identify which current customers are most likely to have issues or complaints in the next week, and proactively suggest what we should do to prevent problems",
                "description": "Tests predictive analysis, risk modeling, proactive service, and prevention strategies",
                "difficulty": "Expert"
            }
        ]

        print("\n🎯 Available Predefined Scenarios:")
        print("=" * 60)

        # Group scenarios by difficulty
        difficulty_groups = {}
        for scenario in scenarios:
            difficulty = scenario['difficulty']
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(scenario)

        # Display scenarios grouped by difficulty
        scenario_index = 1
        for difficulty in ['Basic', 'Intermediate', 'Advanced', 'Expert']:
            if difficulty in difficulty_groups:
                print(f"\n🔸 {difficulty.upper()} SCENARIOS:")
                for scenario in difficulty_groups[difficulty]:
                    print(f"{scenario_index:2d}. {scenario['name']}")
                    print(f"    Query: '{scenario['query'][:80]}{'...' if len(scenario['query']) > 80 else ''}'")
                    print(f"    Description: {scenario['description']}")
                    print()
                    scenario_index += 1

        while True:
            print(f"💡 Options:")
            print(f"   • Enter scenario number (1-{len(scenarios)})")
            print(f"   • 'basic' for scenarios 1-3")
            print(f"   • 'intermediate' for scenarios 4-7")
            print(f"   • 'advanced' for scenarios 8-11")
            print(f"   • 'expert' for scenarios 12-15")
            print(f"   • 'all' for all scenarios")
            print(f"   • 'random' for a random challenging scenario")
            print(f"   • 'back' to return to main menu")

            choice = input("\nSelect option: ").strip().lower()

            if choice == 'back':
                break
            elif choice == 'all':
                for i, scenario in enumerate(scenarios, 1):
                    print(f"\n🔍 Running Scenario {i}: {scenario['name']} ({scenario['difficulty']})")
                    print("=" * 70)
                    self.run_query_with_options(scenario['query'])
                    if i < len(scenarios):
                        cont = input(f"\nPress Enter to continue to next scenario (or 'stop' to finish): ").strip().lower()
                        if cont == 'stop':
                            break
                break
            elif choice == 'basic':
                self._run_difficulty_scenarios(scenarios, 'Basic')
                break
            elif choice == 'intermediate':
                self._run_difficulty_scenarios(scenarios, 'Intermediate')
                break
            elif choice == 'advanced':
                self._run_difficulty_scenarios(scenarios, 'Advanced')
                break
            elif choice == 'expert':
                self._run_difficulty_scenarios(scenarios, 'Expert')
                break
            elif choice == 'random':
                import random
                # Bias towards more challenging scenarios
                weights = [1 if s['difficulty'] == 'Basic' else
                           2 if s['difficulty'] == 'Intermediate' else
                           3 if s['difficulty'] == 'Advanced' else 4
                           for s in scenarios]
                scenario = random.choices(scenarios, weights=weights)[0]
                print(f"\n🎲 Random Scenario: {scenario['name']} ({scenario['difficulty']})")
                print("=" * 70)
                self.run_query_with_options(scenario['query'])
                break
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(scenarios):
                    scenario = scenarios[idx]
                    print(f"\n🔍 Running Scenario: {scenario['name']} ({scenario['difficulty']})")
                    print("=" * 70)
                    self.run_query_with_options(scenario['query'])
                    break
                else:
                    print(f"❌ Invalid scenario number. Please select 1-{len(scenarios)}.")
            else:
                print("❌ Invalid choice. Please try again.")

    def _run_difficulty_scenarios(self, scenarios, difficulty):
        """Run all scenarios of a specific difficulty level"""
        difficulty_scenarios = [s for s in scenarios if s['difficulty'] == difficulty]

        print(f"\n🎯 Running All {difficulty.upper()} Scenarios ({len(difficulty_scenarios)} scenarios)")
        print("=" * 60)

        for i, scenario in enumerate(difficulty_scenarios, 1):
            print(f"\n🔍 {difficulty} Scenario {i}: {scenario['name']}")
            print("=" * 50)
            self.run_query_with_options(scenario['query'])

            if i < len(difficulty_scenarios):
                cont = input(f"\nPress Enter to continue to next {difficulty.lower()} scenario (or 'stop' to finish): ").strip().lower()
                if cont == 'stop':
                    break

    def run_custom_query(self):
        """Run a custom user query"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        print("\n💬 Custom Query Mode")
        print("=" * 30)
        print("You can ask questions about:")
        print("- Order details (e.g., 'What items are in order 1001?')")
        print("- Delivery dates (e.g., 'When will order 1002 be delivered?')")
        print("- Return policies (e.g., 'What's the return policy for laptops?')")
        print("- Customer support information")
        print("- Multi-order analysis (e.g., 'Compare orders 1007, 1017, 1023')")
        print("- Geographic analysis (e.g., 'Orders in Auburn, AL')")
        print("- Predictive analysis (e.g., 'Which customers might have issues?')")
        print("- Any combination of the above")
        print("\nType 'back' to return to main menu.")
        print()

        while True:
            query = input("🤔 Your question: ").strip()

            if query.lower() == 'back':
                break
            elif query:
                self.run_query_with_options(query)
                print("\n" + "-" * 50)
            else:
                print("❌ Please enter a valid question.")

    def run_query(self, query: str):
        """Execute a query using the agent with enhanced error handling"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        try:
            print(f"🤖 Processing query: '{query}'")
            print("-" * 50)

            # Add timeout and retry logic
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Query execution timeout")

            # Set timeout to 60 seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            try:
                response = self.agent.query(query)
                signal.alarm(0)  # Cancel timeout

                print("\n✅ Agent Response:")
                print("=" * 30)
                print(response)
                print()

            except TimeoutError:
                signal.alarm(0)
                print("\n⏰ Query timeout! The query took too long to process.")
                print("💡 Try simplifying your question or breaking it into parts.")

            except Exception as query_error:
                signal.alarm(0)
                print(f"\n❌ Query execution error: {query_error}")

                # Try to provide helpful fallback
                if "max iterations" in str(query_error).lower():
                    print("\n🔄 The query was too complex and reached iteration limit.")
                    print("💡 Suggestions:")
                    print("   • Break down your question into simpler parts")
                    print("   • Ask about specific order IDs or customers")
                    print("   • Try using more specific keywords")
                elif "tool" in str(query_error).lower():
                    print("\n🛠️  Tool execution issue detected.")
                    print("💡 The system might need specific data to answer your question.")
                    print("   • Ensure order IDs, emails, or product names are correct")
                    print("   • Try asking about existing orders (1001-1030)")

        except Exception as e:
            print(f"❌ Critical error during query execution: {e}")
            print("🔧 Please check your database connection and try again.")

    def show_database_stats(self):
        """Show database statistics"""
        print("\n📊 Database Statistics")
        print("=" * 30)

        # Show connection info
        conn_info = self.db_manager.get_connection_info()
        print(f"Connection Type: {conn_info['type']}")
        print(f"Status: {conn_info['status']}")
        if conn_info['type'] == 'connection_pool':
            print(f"Pool Size: {conn_info.get('pool_size', 'Unknown')}")

        stats_queries = [
            ("Total Orders", "SELECT COUNT(*) FROM orders"),
            ("Total Customers", "SELECT COUNT(*) FROM customers"),
            ("Total Products", "SELECT COUNT(*) FROM products"),
            ("Orders by Status", "SELECT status, COUNT(*) FROM orders GROUP BY status"),
            ("Recent Orders (Last 10)", """
                                        SELECT order_id, customer_id, order_date, status, total_amount
                                        FROM orders
                                        ORDER BY order_date DESC
                                            LIMIT 10
                                        """)
        ]

        for stat_name, query in stats_queries:
            print(f"\n{stat_name}:")
            results = self.db_manager.execute_query(query)
            if results:
                if stat_name == "Recent Orders (Last 10)":
                    for row in results:
                        print(f"  Order {row[0]}: Customer {row[1]}, {row[2]}, {row[3]}, ${row[4]}")
                elif stat_name == "Orders by Status":
                    for row in results:
                        print(f"  {row[0]}: {row[1]} orders")
                else:
                    print(f"  {results[0][0]}")

    def main_menu(self):
        """Main interactive menu"""
        if not self.initialize():
            print("❌ Initialization failed. Exiting.")
            return

        while True:
            print("\n" + "=" * 60)
            print("🤖 CUSTOMER SERVICE AI AGENT")
            print("=" * 60)
            print("1. 🎯 Run Predefined Scenarios")
            print("2. 💬 Custom Query")
            print("3. 📊 Database Statistics")
            print("4. 🚪 Exit")
            print("=" * 60)

            choice = input("Select an option (1-4): ").strip()

            if choice == '1':
                self.run_predefined_scenarios()
            elif choice == '2':
                self.run_custom_query()
            elif choice == '3':
                self.show_database_stats()
            elif choice == '4':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")

        # Cleanup
        print("🧹 Cleaning up resources...")
        self.db_manager.disconnect()
        if hasattr(self, 'async_tools') and self.async_tools:
            self.async_tools.close()
            print("✅ Async resources cleaned up")

    def setup_query_decomposer(self):
        """Setup intelligent query decomposition"""
        if self.agent:
            self.query_decomposer = QueryDecomposer(self.agent)
            return True
        return False

    def run_intelligent_query(self, query: str):
        """Enhanced query execution with automatic decomposition"""
        if not hasattr(self, 'query_decomposer') or self.query_decomposer is None:
            print("⚠️  Query decomposer not initialized, using standard execution...")
            return self.run_query(query)

        try:
            print(f"🤖 Processing query with intelligent decomposition: '{query}'")
            print("=" * 70)

            result = self.query_decomposer.execute_decomposed_query(query)

            print("\n✅ Complete Response:")
            print("=" * 50)
            print(result)
            print()

        except Exception as e:
            print(f"❌ Intelligent query execution failed: {e}")
            print("🔄 Falling back to standard execution...")
            self.run_query(query)

    def run_query_with_options(self, query: str):
        """Give user choice between standard and intelligent execution"""
        if not hasattr(self, 'query_decomposer') or self.query_decomposer is None:
            return self.run_query(query)

        # Quick complexity check
        assessment = self.query_decomposer.assess_query_complexity(query)

        if assessment['is_complex']:
            print(f"🧩 Complex query detected (score: {assessment['complexity_score']})")
            print(f"📋 Complexity indicators: {', '.join(assessment['indicators'])}")
            print("\n📋 Execution options:")
            print("1. 🚀 Standard execution (single attempt)")
            print("2. 🧩 Intelligent decomposition (break into subgoals)")
            print("3. 🤖 Auto-decide based on complexity")

            choice = input("Select execution method (1-3, default=3): ").strip() or "3"

            if choice == "1":
                self.run_query(query)
            elif choice == "2":
                self.run_intelligent_query(query)
            else:  # Auto-decide
                if assessment['requires_decomposition']:
                    print("🧩 Auto-selected: Intelligent decomposition")
                    self.run_intelligent_query(query)
                else:
                    print("🚀 Auto-selected: Standard execution")
                    self.run_query(query)
        else:
            print("✅ Simple query detected, using standard execution...")
            self.run_query(query)

    def get_performance_metrics(self):
        """Get performance metrics for the current session"""
        metrics = {
            'database_type': 'Unknown',
            'embedding_device': 'Unknown',
            'tools_count': len(self.tools) if self.tools else 0,
            'parallel_processing': hasattr(self, 'async_tools') and self.async_tools is not None
        }

        # Database metrics
        if hasattr(self, 'db_manager'):
            conn_info = self.db_manager.get_connection_info()
            metrics['database_type'] = conn_info.get('type', 'Unknown')

        # LLM metrics
        if hasattr(Settings, 'embed_model'):
            try:
                device = getattr(Settings.embed_model, 'device', 'Unknown')
                metrics['embedding_device'] = str(device)
            except:
                metrics['embedding_device'] = 'CPU (fallback)'

        return metrics

    def __del__(self):
        """Cleanup async resources and database connections"""
        try:
            if hasattr(self, 'async_tools') and self.async_tools:
                self.async_tools.close()
        except:
            pass

        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                self.db_manager.disconnect()
        except:
            pass