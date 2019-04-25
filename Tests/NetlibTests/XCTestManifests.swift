import XCTest

#if !os(macOS)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(test_DataMigration.allTests),
        testCase(test_IterateView.allTests),
//        testCase(test_StreamOps.allTests),
        testCase(test_Syntax.allTests),
    ]
}
#endif
