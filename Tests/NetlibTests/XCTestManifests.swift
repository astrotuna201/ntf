import XCTest

#if !os(macOS)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(test_DataShape.allTests),
        testCase(test_Codable.allTests),
        testCase(test_Ops.allTests)
    ]
}
#endif
