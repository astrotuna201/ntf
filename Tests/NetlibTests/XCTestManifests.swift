import XCTest

#if !os(macOS)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(NetlibTests.allTests),
        testCase(CodableTests.allTests)
    ]
}
#endif
