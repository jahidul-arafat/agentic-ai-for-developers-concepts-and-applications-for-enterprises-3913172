// Testing Tab Component - COMPLETE UPDATED VERSION WITH PERSISTENT STATE
function TestingTab({ onUpdate, testState, updateTestState }) {
    const {
        testType,
        running,
        results,
        currentTest,
        testProgress,
        progressStats,
        showQuerySelection,
        selectedQueries,
        customQuery,
        customCategory,
        allQueries
    } = testState;

    // Initialize queries when component mounts or test type changes
    useEffect(() => {
        if (allQueries.length === 0 || testType !== testState.testType) {
            const queries = getTestCases(testType);
            updateTestState({
                allQueries: queries,
                selectedQueries: queries.map((_, index) => index),
                testType: testType
            });
        }
    }, [testType]);

    // Enhanced test cases matching the notebook
    const getTestCases = (type) => {
        if (type === 'basic') {
            return [
                { query: "What are the key features?", category: "Generic" },
                { query: "Tell me about specifications", category: "Generic" },
                { query: "What is the warranty coverage?", category: "Generic" }
            ];
        } else {
            let testCases = [
                // Explicit Vehicle Mentions
                { query: "What colors are available for AeroFlow?", category: "Explicit - AeroFlow" },
                { query: "Tell me about EcoSprint's battery specifications", category: "Explicit - EcoSprint" },
                { query: "How do I maintain my AeroFlow vehicle?", category: "Explicit - AeroFlow" },
                { query: "What is EcoSprint's top speed?", category: "Explicit - EcoSprint" },

                // Ambiguous/Comparison Queries
                { query: "Which vehicle has better performance?", category: "Ambiguous - Comparison" },
                { query: "What are the available color options?", category: "Ambiguous - Generic" },
                { query: "Compare the two electric vehicles", category: "Ambiguous - Comparison" },
                { query: "Which one is more environmentally friendly?", category: "Ambiguous - Environmental" },

                // Contextual Keywords
                { query: "Tell me about the eco-friendly features", category: "Contextual - Eco" },
                { query: "What about aerodynamic design?", category: "Contextual - Aero" },
                { query: "How green is this vehicle?", category: "Contextual - Green" },
                { query: "What about the flow dynamics?", category: "Contextual - Flow" },

                // Technical Specifications
                { query: "What is the battery capacity?", category: "Technical - Battery" },
                { query: "How long does charging take?", category: "Technical - Charging" },
                { query: "What safety features are included?", category: "Technical - Safety" },
                { query: "What is the warranty coverage?", category: "Technical - Warranty" },

                // Additional edge cases
                { query: "Compare AeroFlow and EcoSprint features", category: "Explicit Comparison" },
                { query: "Which has better range?", category: "Comparison - Range" },
                { query: "What are the maintenance requirements?", category: "Generic - Maintenance" },
                { query: "Tell me about pricing", category: "Generic - Pricing" },

                // Contextual variations
                { query: "What about the family-friendly features?", category: "Contextual - Family" },
                { query: "How efficient is the motor?", category: "Technical - Efficiency" },
                { query: "What charging options are available?", category: "Technical - Charging Options" },
                { query: "Tell me about the interior design", category: "Technical - Design" }
            ];

            return testCases;
        }
    };

    // Check for JSON files helper
    const hasJSONFiles = (files) => {
        return files && files.some(file =>
            file.original_name && file.original_name.toLowerCase().endsWith('.json')
        );
    };

    // Get JSON-specific test cases
    const getJSONTestCases = () => {
        return [
            { query: "What data is contained in the JSON files?", category: "JSON - Content" },
            { query: "Analyze the structure of the uploaded JSON data", category: "JSON - Structure" },
            { query: "What are the key fields in the JSON data?", category: "JSON - Schema" },
            { query: "Show me statistics from the JSON data", category: "JSON - Statistics" },
            { query: "What patterns can you find in the JSON data?", category: "JSON - Analysis" },
            { query: "How many records are in the JSON dataset?", category: "JSON - Count" },
            { query: "What is the data type distribution in the JSON?", category: "JSON - Types" },
            { query: "Extract insights from the JSON data", category: "JSON - Insights" }
        ];
    };

    // Helper function to determine expected route
    const getExpectedRoute = (testCase) => {
        const query = testCase.query.toLowerCase();
        const category = testCase.category.toLowerCase();

        if (category.includes('aeroflow') || query.includes('aeroflow')) {
            return {
                name: 'AeroFlow',
                color: 'bg-purple-100 text-purple-800',
                icon: '🚁'
            };
        } else if (category.includes('ecosprint') || query.includes('ecosprint')) {
            return {
                name: 'EcoSprint',
                color: 'bg-green-100 text-green-800',
                icon: '🌱'
            };
        } else if (category.includes('comparison') || query.includes('compare') || query.includes('better') || query.includes('vs')) {
            return {
                name: 'Comparison',
                color: 'bg-yellow-100 text-yellow-800',
                icon: '⚖️'
            };
        } else if (category.includes('aero') || query.includes('aerodynamic')) {
            return {
                name: 'AeroFlow',
                color: 'bg-purple-100 text-purple-800',
                icon: '🚁'
            };
        } else if (category.includes('eco') || query.includes('eco') || query.includes('green') || query.includes('environment')) {
            return {
                name: 'EcoSprint',
                color: 'bg-green-100 text-green-800',
                icon: '🌱'
            };
        } else if (category.includes('json')) {
            return {
                name: 'JSON Data',
                color: 'bg-blue-100 text-blue-800',
                icon: '📊'
            };
        } else {
            return {
                name: 'Generic',
                color: 'bg-gray-100 text-gray-800',
                icon: '❓'
            };
        }
    };

    const runTestsWithProgress = async () => {
        updateTestState({
            running: true,
            results: null,
            testProgress: [],
            currentTest: null
        });

        let testCases = selectedQueries.map(index => allQueries[index]);

        if (testCases.length === 0) {
            alert('Please select at least one test query to execute.');
            updateTestState({ running: false });
            return;
        }

        // Check if JSON files are uploaded and add JSON tests if they exist
        if (testType === 'comprehensive' && selectedQueries.length === allQueries.length) {
            try {
                const response = await fetch('/api/files');
                const data = await response.json();

                if (hasJSONFiles(data.files)) {
                    const jsonTests = getJSONTestCases();
                    testCases = [...testCases, ...jsonTests];
                    console.log('📊 Added JSON-specific tests - JSON files detected');
                }
            } catch (error) {
                console.warn('Could not check for JSON files:', error);
            }
        }

        updateTestState({ progressStats: { completed: 0, total: testCases.length } });

        const progressResults = [];

        for (let i = 0; i < testCases.length; i++) {
            const testCase = testCases[i];

            // Update current test
            updateTestState({
                currentTest: {
                    index: i + 1,
                    total: testCases.length,
                    query: testCase.query,
                    category: testCase.category,
                    status: 'running'
                }
            });

            try {
                // Execute individual test
                const startTime = Date.now();
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: testCase.query })
                });

                const data = await response.json();
                const endTime = Date.now();
                const responseTime = (endTime - startTime) / 1000;

                // Enhanced routing intelligence extraction
                let routingIntelligence = data.routing_intelligence || data.routing_info || {};

                if (!routingIntelligence.decision && data.response) {
                    const responseText = data.response.toLowerCase();
                    if (responseText.includes('ecosprint')) {
                        routingIntelligence = {
                            decision: 'EcoSprint_specifications',
                            method_used: 'LLM',
                            reasoning: 'Inferred from response content mentioning EcoSprint'
                        };
                    } else if (responseText.includes('aeroflow')) {
                        routingIntelligence = {
                            decision: 'AeroFlow_specifications',
                            method_used: 'LLM',
                            reasoning: 'Inferred from response content mentioning AeroFlow'
                        };
                    }
                }

                const testResult = {
                    test_id: i + 1,
                    query: testCase.query,
                    category: testCase.category,
                    success: data.success,
                    response: data.response || '',
                    response_time: responseTime,
                    response_length: data.response ? data.response.length : 0,
                    timestamp: new Date().toISOString(),
                    routing_intelligence: routingIntelligence,
                    error: data.error || null
                };

                progressResults.push(testResult);

                // Update progress - USE FUNCTIONAL UPDATES TO AVOID STALE STATE
                updateTestState(prev => ({
                    testProgress: [...prev.testProgress, testResult],
                    progressStats: { completed: i + 1, total: testCases.length },
                    currentTest: { ...prev.currentTest, status: 'completed', success: data.success }
                }));

                // Small delay to show progress
                await new Promise(resolve => setTimeout(resolve, 500));

            } catch (error) {
                const testResult = {
                    test_id: i + 1,
                    query: testCase.query,
                    category: testCase.category,
                    success: false,
                    error: error.message,
                    timestamp: new Date().toISOString()
                };

                progressResults.push(testResult);

                updateTestState(prev => ({
                    testProgress: [...prev.testProgress, testResult],
                    progressStats: { completed: i + 1, total: testCases.length },
                    currentTest: { ...prev.currentTest, status: 'failed', success: false }
                }));

                await new Promise(resolve => setTimeout(resolve, 500));
            }
        }

        // Calculate final results
        const successful = progressResults.filter(r => r.success);
        const summary = {
            total_tests: progressResults.length,
            successful_tests: successful.length,
            success_rate: (successful.length / progressResults.length) * 100,
            average_response_time: successful.length > 0
                ? successful.reduce((sum, r) => sum + (r.response_time || 0), 0) / successful.length
                : 0,
            average_response_length: successful.length > 0
                ? successful.reduce((sum, r) => sum + (r.response_length || 0), 0) / successful.length
                : 0
        };

        const finalResults = {
            test_type: testType,
            timestamp: new Date().toISOString(),
            summary: summary,
            results: progressResults
        };

        updateTestState({
            results: finalResults,
            currentTest: null,
            running: false
        });

        onUpdate();
    };

    return (
        <div className="space-y-6">
            {/* Test Query Selection */}
            <div className="bg-white shadow rounded-lg p-6">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-lg font-medium text-gray-900">Test Query Selection</h2>
                    <button
                        onClick={() => updateTestState({ showQuerySelection: !showQuerySelection })}
                        className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                    >
                        {showQuerySelection ? '📝 Hide Query Selection' : '📝 Customize Queries'}
                    </button>
                </div>

                {showQuerySelection && (
                    <div className="space-y-6">
                        {/* Query List */}
                        <div>
                            <div className="flex justify-between items-center mb-3">
                                <h3 className="text-md font-medium text-gray-800">
                                    Available Test Queries ({allQueries.length})
                                </h3>
                                <div className="flex space-x-2">
                                    <button
                                        onClick={() => updateTestState({ selectedQueries: allQueries.map((_, index) => index) })}
                                        className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded hover:bg-blue-200"
                                    >
                                        Select All
                                    </button>
                                    <button
                                        onClick={() => updateTestState({ selectedQueries: [] })}
                                        className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded hover:bg-gray-200"
                                    >
                                        Clear All
                                    </button>
                                </div>
                            </div>

                            <div className="max-h-80 overflow-y-auto border border-gray-200 rounded-md">
                                <table className="min-w-full bg-white">
                                    <thead className="bg-gray-50 sticky top-0">
                                    <tr>
                                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            <input
                                                type="checkbox"
                                                checked={selectedQueries.length === allQueries.length}
                                                onChange={(e) => {
                                                    if (e.target.checked) {
                                                        updateTestState({ selectedQueries: allQueries.map((_, index) => index) });
                                                    } else {
                                                        updateTestState({ selectedQueries: [] });
                                                    }
                                                }}
                                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                            />
                                        </th>
                                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Category
                                        </th>
                                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Query
                                        </th>
                                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Expected Route
                                        </th>
                                    </tr>
                                    </thead>
                                    <tbody className="bg-white divide-y divide-gray-200">
                                    {allQueries.map((testCase, index) => {
                                        const isSelected = selectedQueries.includes(index);
                                        const expectedRoute = getExpectedRoute(testCase);

                                        return (
                                            <tr
                                                key={index}
                                                className={`${isSelected ? 'bg-blue-50' : 'hover:bg-gray-50'} cursor-pointer`}
                                                onClick={() => {
                                                    if (isSelected) {
                                                        updateTestState({
                                                            selectedQueries: selectedQueries.filter(i => i !== index)
                                                        });
                                                    } else {
                                                        updateTestState({
                                                            selectedQueries: [...selectedQueries, index]
                                                        });
                                                    }
                                                }}
                                            >
                                                <td className="px-3 py-3 whitespace-nowrap">
                                                    <input
                                                        type="checkbox"
                                                        checked={isSelected}
                                                        onChange={() => {}} // Handled by row click
                                                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                                    />
                                                </td>
                                                <td className="px-3 py-3 whitespace-nowrap">
                                                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                                                            {testCase.category}
                                                        </span>
                                                </td>
                                                <td className="px-3 py-3 text-sm text-gray-900 max-w-md">
                                                    <div className="truncate" title={testCase.query}>
                                                        {testCase.query}
                                                    </div>
                                                </td>
                                                <td className="px-3 py-3 whitespace-nowrap">
                                                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${expectedRoute.color}`}>
                                                            {expectedRoute.icon} {expectedRoute.name}
                                                        </span>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Add Custom Query */}
                        <div className="border-t pt-4">
                            <h4 className="text-md font-medium text-gray-800 mb-3">Add Custom Query</h4>
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                                <div className="sm:col-span-2">
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Custom Query
                                    </label>
                                    <input
                                        type="text"
                                        value={customQuery}
                                        onChange={(e) => updateTestState({ customQuery: e.target.value })}
                                        placeholder="Enter your custom test query..."
                                        className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Category
                                    </label>
                                    <input
                                        type="text"
                                        value={customCategory}
                                        onChange={(e) => updateTestState({ customCategory: e.target.value })}
                                        placeholder="Category"
                                        className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                            </div>
                            <button
                                onClick={() => {
                                    if (customQuery.trim()) {
                                        const newQuery = {
                                            query: customQuery.trim(),
                                            category: customCategory.trim() || 'Custom'
                                        };
                                        updateTestState({
                                            allQueries: [...allQueries, newQuery],
                                            selectedQueries: [...selectedQueries, allQueries.length],
                                            customQuery: '',
                                            customCategory: 'Custom'
                                        });
                                    }
                                }}
                                disabled={!customQuery.trim()}
                                className="mt-2 bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                            >
                                ➕ Add Query
                            </button>
                        </div>

                        {/* Selection Summary */}
                        <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="flex justify-between items-center">
                                <div>
                                    <span className="text-sm font-medium text-blue-900">
                                        {selectedQueries.length} of {allQueries.length} queries selected
                                    </span>
                                    {selectedQueries.length > 0 && (
                                        <div className="text-xs text-blue-700 mt-1">
                                            Categories: {[...new Set(selectedQueries.map(i => allQueries[i].category))].join(', ')}
                                        </div>
                                    )}
                                </div>
                                <div className="text-xs text-blue-600">
                                    Est. time: ~{(selectedQueries.length * 20).toFixed(0)} seconds
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Test Configuration */}
            <div className="bg-white shadow rounded-lg p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-6">Router Testing</h2>

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Test Type
                        </label>
                        <div className="space-y-2">
                            <label className="flex items-center">
                                <input
                                    type="radio"
                                    value="basic"
                                    checked={testType === 'basic'}
                                    onChange={(e) => updateTestState({ testType: e.target.value })}
                                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                    disabled={running}
                                />
                                <span className="ml-2 text-sm text-gray-900">
                                    <strong>Basic Test</strong> - Simple functionality check (3 tests)
                                </span>
                            </label>
                            <label className="flex items-center">
                                <input
                                    type="radio"
                                    value="comprehensive"
                                    checked={testType === 'comprehensive'}
                                    onChange={(e) => updateTestState({ testType: e.target.value })}
                                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                    disabled={running}
                                />
                                <span className="ml-2 text-sm text-gray-900">
                                    <strong>Comprehensive Test</strong> - Full routing intelligence test (24+ tests)
                                </span>
                            </label>
                        </div>
                    </div>

                    <button
                        onClick={runTestsWithProgress}
                        disabled={running || selectedQueries.length === 0}
                        className="bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                    >
                        {running ? (
                            <>
                                <span className="spinner mr-2"></span>
                                Running Tests...
                            </>
                        ) : (
                            `🚀 Run Selected Tests (${selectedQueries.length})`
                        )}
                    </button>
                </div>
            </div>

            {/* Real-time Test Progress */}
            {running && (
                <div className="bg-white shadow rounded-lg p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Test Progress</h3>

                    {/* Progress Bar */}
                    <div className="mb-4">
                        <div className="flex justify-between text-sm text-gray-600 mb-1">
                            <span>Progress</span>
                            <span>{progressStats.completed}/{progressStats.total} tests completed</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${(progressStats.completed / progressStats.total) * 100}%` }}
                            ></div>
                        </div>
                    </div>

                    {/* Current Test */}
                    {currentTest && (
                        <div className="mb-4 p-4 border border-blue-200 rounded-lg bg-blue-50">
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="font-medium text-blue-900">
                                    Test {currentTest.index}/{currentTest.total}
                                </h4>
                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                    currentTest.status === 'running'
                                        ? 'bg-yellow-100 text-yellow-800'
                                        : currentTest.success
                                            ? 'bg-green-100 text-green-800'
                                            : 'bg-red-100 text-red-800'
                                }`}>
                                    {currentTest.status === 'running' && <span className="spinner mr-1"></span>}
                                    {currentTest.status === 'running' ? 'Running...'
                                        : currentTest.status === 'completed' && currentTest.success ? '✅ Passed'
                                            : currentTest.status === 'completed' && !currentTest.success ? '❌ Failed'
                                                : '⏳ Pending'}
                                </span>
                            </div>
                            <div className="text-sm text-blue-700">
                                <div className="font-medium mb-1">Category: {currentTest.category}</div>
                                <div>Query: "{currentTest.query}"</div>
                            </div>
                        </div>
                    )}

                    {/* Completed Tests Log */}
                    {testProgress.length > 0 && (
                        <div>
                            <h4 className="font-medium text-gray-900 mb-3">Completed Tests</h4>
                            <div className="space-y-2 max-h-60 overflow-y-auto">
                                {testProgress.map((test, index) => (
                                    <div key={index} className="flex items-center justify-between p-3 border border-gray-200 rounded">
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-2">
                                                <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                                                    test.success
                                                        ? 'bg-green-100 text-green-800'
                                                        : 'bg-red-100 text-red-800'
                                                }`}>
                                                    {test.success ? '✅' : '❌'}
                                                </span>
                                                <span className="text-sm text-gray-500">{test.category}</span>
                                            </div>
                                            <div className="text-sm text-gray-900 mt-1 truncate">
                                                {test.query}
                                            </div>
                                        </div>
                                        <div className="text-xs text-gray-500 ml-4">
                                            {test.response_time?.toFixed(2)}s
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Test Results */}
            {results && !running && (
                <div className="bg-white shadow rounded-lg p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Test Results</h3>

                    <div className="space-y-6">
                        {/* Summary */}
                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
                            <div className="bg-blue-50 p-4 rounded-lg">
                                <div className="text-2xl font-bold text-blue-600">
                                    {results.summary.total_tests}
                                </div>
                                <div className="text-sm text-blue-800">Total Tests</div>
                            </div>
                            <div className="bg-green-50 p-4 rounded-lg">
                                <div className="text-2xl font-bold text-green-600">
                                    {results.summary.successful_tests}
                                </div>
                                <div className="text-sm text-green-800">Successful</div>
                            </div>
                            <div className="bg-yellow-50 p-4 rounded-lg">
                                <div className="text-2xl font-bold text-yellow-600">
                                    {results.summary.success_rate.toFixed(1)}%
                                </div>
                                <div className="text-sm text-yellow-800">Success Rate</div>
                            </div>
                            <div className="bg-purple-50 p-4 rounded-lg">
                                <div className="text-2xl font-bold text-purple-600">
                                    {results.summary.average_response_time?.toFixed(2)}s
                                </div>
                                <div className="text-sm text-purple-800">Avg Response</div>
                            </div>
                        </div>

                        {/* Individual Test Results */}
                        <div>
                            {/* Routing Intelligence Summary Table */}
                            <div className="mb-6 overflow-x-auto">
                                <h5 className="font-medium text-gray-900 mb-3">🧠 Routing Intelligence Summary</h5>
                                <table className="min-w-full bg-white border border-gray-200 rounded-lg">
                                    <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Test #</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Query</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Route Chosen</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Method</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reasoning</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                    </tr>
                                    </thead>
                                    <tbody className="bg-white divide-y divide-gray-200">
                                    {results.results.map((result, index) => {
                                        const routingInfo = result.routing_intelligence || {};
                                        const routeChosen = routingInfo.decision || 'Unknown';
                                        const method = routingInfo.method_used || 'Unknown';
                                        const reasoning = routingInfo.final_reasoning ||
                                            routingInfo.reasoning ||
                                            (routingInfo.reasoning_steps && routingInfo.reasoning_steps.join('; ')) ||
                                            'No reasoning available';

                                        // Extract route name (remove "_specifications" suffix if present)
                                        const displayRoute = routeChosen.replace(/_specifications$/, '').replace(/_/g, ' ');

                                        return (
                                            <tr key={index} className={`${result.success ? 'bg-green-50' : 'bg-red-50'} hover:bg-gray-50`}>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                                                    #{result.test_id}
                                                </td>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                                                       <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                                           {result.category}
                                                       </span>
                                                </td>
                                                <td className="px-4 py-3 text-sm text-gray-900 max-w-xs">
                                                    <div className="truncate" title={result.query}>
                                                        {result.query}
                                                    </div>
                                                </td>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                                                       <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                           routeChosen === 'Unknown'
                                                               ? 'bg-gray-100 text-gray-800'
                                                               : routeChosen.toLowerCase().includes('aeroflow')
                                                                   ? 'bg-purple-100 text-purple-800'
                                                                   : routeChosen.toLowerCase().includes('ecosprint')
                                                                       ? 'bg-green-100 text-green-800'
                                                                       : 'bg-yellow-100 text-yellow-800'
                                                       }`}>
                                                           {routeChosen === 'Unknown' ? '❓ Unknown' : `🎯 ${displayRoute}`}
                                                       </span>
                                                </td>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                                                       <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                                                           method === 'LLM'
                                                               ? 'bg-blue-100 text-blue-800'
                                                               : method === 'Keyword'
                                                                   ? 'bg-orange-100 text-orange-800'
                                                                   : 'bg-gray-100 text-gray-800'
                                                       }`}>
                                                           {method === 'LLM' ? '🧠 LLM' : method === 'Keyword' ? '🔍 Keyword' : '❓ Unknown'}
                                                       </span>
                                                </td>
                                                <td className="px-4 py-3 text-sm text-gray-600 max-w-md">
                                                    <div className="truncate" title={reasoning}>
                                                        {reasoning.length > 80 ? reasoning.substring(0, 80) + '...' : reasoning}
                                                    </div>
                                                </td>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm">
                                                       <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                           result.success
                                                               ? 'bg-green-100 text-green-800'
                                                               : 'bg-red-100 text-red-800'
                                                       }`}>
                                                           {result.success ? '✅ Pass' : '❌ Fail'}
                                                       </span>
                                                </td>
                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                                                    {result.response_time?.toFixed(2)}s
                                                </td>
                                            </tr>
                                        );
                                    })}
                                    </tbody>
                                </table>
                            </div>

                            {/* Routing Statistics Summary */}
                            <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-4">
                                <div className="bg-purple-50 p-4 rounded-lg">
                                    <div className="text-2xl font-bold text-purple-600">
                                        {results.results.filter(r => r.routing_intelligence?.decision?.toLowerCase().includes('aeroflow')).length}
                                    </div>
                                    <div className="text-sm text-purple-800">AeroFlow Routes</div>
                                </div>
                                <div className="bg-green-50 p-4 rounded-lg">
                                    <div className="text-2xl font-bold text-green-600">
                                        {results.results.filter(r => r.routing_intelligence?.decision?.toLowerCase().includes('ecosprint')).length}
                                    </div>
                                    <div className="text-sm text-green-800">EcoSprint Routes</div>
                                </div>
                                <div className="bg-blue-50 p-4 rounded-lg">
                                    <div className="text-2xl font-bold text-blue-600">
                                        {results.results.filter(r => r.routing_intelligence?.method_used === 'LLM').length}
                                    </div>
                                    <div className="text-sm text-blue-800">LLM Decisions</div>
                                </div>
                                <div className="bg-orange-50 p-4 rounded-lg">
                                    <div className="text-2xl font-bold text-orange-600">
                                        {results.results.filter(r => r.routing_intelligence?.method_used === 'Keyword').length}
                                    </div>
                                    <div className="text-sm text-orange-800">Keyword Decisions</div>
                                </div>
                            </div>

                            {/* Detailed Test Results (Expandable) */}
                            <details className="mb-4">
                                <summary className="cursor-pointer font-medium text-gray-900 hover:text-blue-600">
                                    📋 View Detailed Test Results ({results.results.length} tests)
                                </summary>
                                <div className="mt-4 space-y-3">
                                    {results.results.map((result, index) => (
                                        <div key={index} className="border border-gray-200 rounded-lg p-4">
                                            <div className="flex items-start justify-between">
                                                <div className="flex-1">
                                                    <div className="flex items-center space-x-2 mb-1">
                                                       <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                           result.success
                                                               ? 'bg-green-100 text-green-800'
                                                               : 'bg-red-100 text-red-800'
                                                       }`}>
                                                           {result.success ? '✅ Pass' : '❌ Fail'}
                                                       </span>
                                                        <span className="text-sm text-gray-500">
                                                           {result.category}
                                                       </span>
                                                        <span className="text-xs text-gray-400">
                                                           Test #{result.test_id}
                                                       </span>
                                                    </div>
                                                    <p className="text-sm font-medium text-gray-900 mb-2">
                                                        {result.query}
                                                    </p>

                                                    {/* Routing Intelligence Details */}
                                                    {result.routing_intelligence && (
                                                        <div className="mb-3 p-3 bg-blue-50 rounded border-l-4 border-blue-400">
                                                            <h6 className="text-sm font-medium text-blue-900 mb-2">🧠 Routing Intelligence</h6>
                                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-blue-800">
                                                                <div>
                                                                    <span className="font-medium">Route:</span> {result.routing_intelligence.decision || 'Unknown'}
                                                                </div>
                                                                <div>
                                                                    <span className="font-medium">Method:</span> {result.routing_intelligence.method_used || 'Unknown'}
                                                                </div>
                                                                {result.routing_intelligence.scores && (
                                                                    <div className="col-span-2">
                                                                        <span className="font-medium">Scores:</span> {JSON.stringify(result.routing_intelligence.scores)}
                                                                    </div>
                                                                )}
                                                                {(result.routing_intelligence.final_reasoning || result.routing_intelligence.reasoning) && (
                                                                    <div className="col-span-2">
                                                                        <span className="font-medium">Reasoning:</span> {result.routing_intelligence.final_reasoning || result.routing_intelligence.reasoning}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    )}

                                                    {result.success ? (
                                                        <div className="text-sm text-gray-600">
                                                            <div className="flex items-center space-x-4 mb-2">
                                                                <span>Response: {result.response_length} chars</span>
                                                                <span>Time: {result.response_time?.toFixed(2)}s</span>
                                                            </div>
                                                            <details className="mt-2">
                                                                <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                                                                    View Response
                                                                </summary>
                                                                <div className="mt-2 p-3 bg-gray-50 rounded text-xs border-l-4 border-blue-400">
                                                                    {result.response}
                                                                </div>
                                                            </details>
                                                        </div>
                                                    ) : (
                                                        <div className="text-sm text-red-600">
                                                            Error: {result.error}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </details>
                        </div>
                    </div>
                </div>
            )}

            {/* Quick Query Test Component */}
            <QuickQueryTest />
        </div>
    );
}

// Quick Query Test Component (remains the same but add it here for completeness)
function QuickQueryTest() {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const predefinedQueries = [
        // Explicit mentions
        "What colors are available for AeroFlow?",
        "Tell me about EcoSprint's battery specifications",
        "How do I maintain my AeroFlow vehicle?",
        "What is EcoSprint's top speed?",

        // Ambiguous queries
        "Which vehicle has better performance?",
        "What are the available color options?",
        "Compare the two electric vehicles",
        "Which one is more environmentally friendly?",

        // Contextual keywords
        "Tell me about the eco-friendly features",
        "What about aerodynamic design?",
        "How green is this vehicle?",
        "What about the flow dynamics?",

        // Technical specs
        "What is the battery capacity?",
        "How long does charging take?",
        "What safety features are included?",
        "What is the warranty coverage?",

        // JSON Contents
        "What data is contained in the JSON files?",
        "Analyze the structure of the uploaded JSON data",
        "What are the key fields in the JSON data?",
        "Show me statistics from the JSON data",
        "What patterns can you find in the JSON data?",
        "How many records are in the JSON dataset?",
        "What is the data type distribution in the JSON?",
        "Extract insights from the JSON data"
    ];

    const executeQuery = async (queryText) => {
        setLoading(true);
        setResult(null);

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: queryText })
            });

            const data = await response.json();
            setResult(data);
        } catch (error) {
            setResult({ success: false, error: error.message });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white shadow rounded-lg p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Quick Query Test</h3>

            <div className="space-y-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Enter Query
                    </label>
                    <div className="flex space-x-2">
                        <input
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Ask a question..."
                            className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            onKeyPress={(e) => e.key === 'Enter' && executeQuery(query)}
                        />
                        <button
                            onClick={() => executeQuery(query)}
                            disabled={loading || !query.trim()}
                            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                        >
                            {loading ? <span className="spinner"></span> : 'Ask'}
                        </button>
                    </div>
                </div>

                <div>
                    <p className="text-sm text-gray-700 mb-2">Or try a predefined query:</p>
                    <div className="flex flex-wrap gap-2">
                        {predefinedQueries.map((predefinedQuery, index) => (
                            <button
                                key={index}
                                onClick={() => executeQuery(predefinedQuery)}
                                disabled={loading}
                                className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded hover:bg-gray-200 disabled:opacity-50"
                            >
                                {predefinedQuery}
                            </button>
                        ))}
                    </div>
                </div>

                {result && (
                    <div className="border-t pt-4">
                        {result.success ? (
                            <div className="space-y-2">
                                <div className="text-sm text-gray-600">
                                    Response time: {result.response_time?.toFixed(2)}s
                                </div>
                                <div className="bg-gray-50 p-3 rounded border">
                                    <p className="text-sm">{result.response}</p>
                                </div>
                                {result.routing_info && (
                                    <details className="text-xs text-gray-500">
                                        <summary className="cursor-pointer">Routing Details</summary>
                                        <pre className="mt-2 p-2 bg-gray-100 rounded">
                                           {JSON.stringify(result.routing_info, null, 2)}
                                       </pre>
                                    </details>
                                )}
                            </div>
                        ) : (
                            <div className="text-red-600 text-sm">
                                Error: {result.error}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}