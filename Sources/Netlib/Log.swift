//******************************************************************************
//  Created by Edward Connell on 4/12/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

//==============================================================================
// Logging
public protocol Logging {
    var log: Log? { get }
	var logLevel: LogLevel { get set }
	var namePath: String { get }
	var nestingLevel: Int { get }
}

//------------------------------------------------------------------------------
// Logging
extension Logging {
	//------------------------------------
	// willLog
	public func willLog(level: LogLevel) -> Bool {
		guard let thisLog = log else { return false }
		return level <= thisLog.logLevel || level <= logLevel
	}

	//------------------------------------
	// writeLog
	public func writeLog(_ message: String, level: LogLevel = .error,
	                     indent: Int = 0, trailing: String = "",
	                     minCount: Int = 80) {
		if willLog(level: level) {
			log!.write(level: level, nestingLevel: indent + nestingLevel,
			           trailing: trailing, minCount: minCount, message: message)
		}
	}

	//------------------------------------
	// diagnostic
	public func diagnostic(_ message: String, categories: LogCategories,
	                       indent: Int = 0, trailing: String = "",
	                       minCount: Int = 80) {
		if willLog(level: .diagnostic) {
			// if subcategories have been selected on the log object
			// then make sure the caller's category is desired
			if let mask = log?.categories?.rawValue,
			   categories.rawValue & mask == 0 { return }

			log!.write(
					level: .diagnostic, nestingLevel: indent + nestingLevel, 
					trailing: trailing, minCount: minCount, message: message)
		}
	}
}

//==============================================================================
// Log
final public class Log : ObjectTracking {
	//--------------------------------------------------------------------------
	// properties
	public var categories: LogCategories?
	public var maxHistory = 0
	public var silent = false
	public var tabSize = 2
	public var url: URL?
	public var history = [LogEvent]()
    public var logLevel = LogLevel.error
    public var nestingLevel = 0
    public var namePath: String

    // ObjectTracking
    public private(set) var trackingId: Int = 0
    
    // A log can be written to freely by any thread,
    // so create write queue
	private let queue = DispatchQueue(label: "Log.queue")
	private static let levelColWidth =
		String(describing: LogLevel.diagnostic).count

	// initializer to allow for Log to be used as a parameter default
	public init(parentNamePath: String) {
		namePath = parentNamePath + ".Log"		
	}
	
	//----------------------------------------------------------------------------
	// functions
	public func write(level: LogLevel, nestingLevel: Int,
	                  trailing: String, minCount: Int, message: String) {
		queue.sync() { [unowned self] in
			if self.maxHistory > 0 {
				if self.history.count == self.maxHistory { self.history.removeFirst() }
				self.history.append(LogEvent(level: level, nestingLevel: nestingLevel,
				                             message: message))
			}

			let levelStr = String(describing: level).padding(
				toLength: Log.levelColWidth, withPad: " ", startingAt: 0)

			let indent = String(repeating: " ", count: nestingLevel * self.tabSize)
			var eventStr = levelStr + ": " + indent + message

			// add trailing fill
			if !trailing.isEmpty {
				let fillCount = minCount - eventStr.count
				if message.isEmpty {
					eventStr += String(repeating: trailing, count: fillCount)
				} else {
					if fillCount > 1 {
						eventStr += " " + String(repeating: trailing, count: fillCount - 1)
					}
				}
			}

			// TODO: add write to log file support
			//		if let uri = uri {
			//
			//		}

			// write to the console
			if !self.silent && self.url == nil && self.maxHistory == 0 {
				print(eventStr)
			}
		}
	}
}

//==============================================================================
// LogEvent
public struct LogEvent {
	var level       : LogLevel
	var nestingLevel: Int
	var message     : String
}

//------------------------------------------------------------------------------
// LogColors
//  http://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
public enum LogColor : String {
	case reset       = "\u{1b}[m"
	case red         = "\u{1b}[31m"
	case green       = "\u{1b}[32m"
	case yellow      = "\u{1b}[33m"
	case blue        = "\u{1b}[34m"
	case magenta     = "\u{1b}[35m"
	case cyan        = "\u{1b}[36m"
	case white       = "\u{1b}[37m"
	case bold        = "\u{1b}[1m"
	case boldRed     = "\u{1b}[1;31m"
	case boldGreen   = "\u{1b}[1;32m"
	case boldYellow  = "\u{1b}[1;33m"
	case boldBlue    = "\u{1b}[1;34m"
	case boldMagenta = "\u{1b}[1;35m"
	case boldCyan    = "\u{1b}[1;36m"
	case boldWhite   = "\u{1b}[1;37m"
}

public func setText(_ text: String, color: LogColor) -> String {
	#if os(Linux)
	return color.rawValue + text + LogColor.reset.rawValue
	#else
	return text
	#endif
}

//------------------------------------------------------------------------------
// LogCategories
public struct LogCategories: OptionSet {
	public init(rawValue: Int) { self.rawValue = rawValue }

	public init?(string: String) throws {
		var value = 0
		let options = string.components(separatedBy: ",")
		for i in 0..<options.count {
			let option = options[i].trimmingCharacters(in: .whitespacesAndNewlines)
			switch option {
			case "connections"  : value |= LogCategories.connections.rawValue
			case "context"      : value |= LogCategories.context.rawValue
			case "dataAlloc"    : value |= LogCategories.dataAlloc.rawValue
			case "dataCopy"     : value |= LogCategories.dataCopy.rawValue
			case "dataMutation" : value |= LogCategories.dataMutation.rawValue
			case "defaults"     : value |= LogCategories.defaultsLookup.rawValue
			case "evaluate"     : value |= LogCategories.evaluate.rawValue
			case "setup"        : value |= LogCategories.setup.rawValue
			case "setupBackward": value |= LogCategories.setupBackward.rawValue
			case "setupForward" : value |= LogCategories.setupForward.rawValue
			case "streamAlloc"  : value |= LogCategories.streamAlloc.rawValue
			case "streamSync"   : value |= LogCategories.streamSync.rawValue
			case "tryLookup"    : value |= LogCategories.tryDefaultsLookup.rawValue
			case "download"     : value |= LogCategories.download.rawValue
			default: return nil
			}
		}
		self.rawValue = value
	}

	public var string: String {
		var result = ""
		if rawValue & LogCategories.connections.rawValue    != 0 { result += "connections, " }
		if rawValue & LogCategories.context.rawValue        != 0 { result += "context, " }
		if rawValue & LogCategories.dataAlloc.rawValue      != 0 { result += "dataAlloc, " }
		if rawValue & LogCategories.dataCopy.rawValue       != 0 { result += "dataCopy, " }
		if rawValue & LogCategories.dataMutation.rawValue   != 0 { result += "dataMutation, " }
		if rawValue & LogCategories.defaultsLookup.rawValue != 0 { result += "defaults, " }
		if rawValue & LogCategories.evaluate.rawValue       != 0 { result += "evaluate, " }
		if rawValue & LogCategories.setup.rawValue          != 0 { result += "setup, " }
		if rawValue & LogCategories.setupBackward.rawValue  != 0 { result += "setupBackward, " }
		if rawValue & LogCategories.setupForward.rawValue   != 0 { result += "setupForward, " }
		if rawValue & LogCategories.streamAlloc.rawValue    != 0 { result += "streamAlloc, " }
		if rawValue & LogCategories.streamSync.rawValue     != 0 { result += "streamSync, " }
		if rawValue & LogCategories.tryDefaultsLookup.rawValue != 0 { result += "tryLookup, " }
		if rawValue & LogCategories.download.rawValue       != 0 { result += "download, " }
		if !result.isEmpty { result.removeLast(2) }
		return result
	}

	// properties
	public let rawValue: Int
	public static let connections       = LogCategories(rawValue: 1 << 0)
	public static let dataAlloc         = LogCategories(rawValue: 1 << 1)
	public static let dataCopy          = LogCategories(rawValue: 1 << 2)
	public static let dataMutation      = LogCategories(rawValue: 1 << 3)
	public static let defaultsLookup    = LogCategories(rawValue: 1 << 4)
	public static let evaluate          = LogCategories(rawValue: 1 << 5)
	public static let setup             = LogCategories(rawValue: 1 << 7)
	public static let setupBackward     = LogCategories(rawValue: 1 << 8)
	public static let setupForward      = LogCategories(rawValue: 1 << 9)
	public static let streamAlloc       = LogCategories(rawValue: 1 << 10)
	public static let streamSync        = LogCategories(rawValue: 1 << 11)
	public static let context           = LogCategories(rawValue: 1 << 12)
	public static let tryDefaultsLookup = LogCategories(rawValue: 1 << 13)
	public static let download          = LogCategories(rawValue: 1 << 14)
}

// strings
let createString   = "[\(setText("CREATE ", color: .cyan))]"
let copyString     = "[\(setText("COPY   ", color: .blue))]"
let releaseString  = "[\(setText("RELEASE", color: .cyan))]"
let blockString    = "[\(setText("BLOCK  ", color: .red))]"
let waitString     = "[\(setText("WAIT   ", color: .yellow))]"
let syncString     = "[\(setText("SYNC   ", color: .yellow))]"
let recordString   = "[\(setText("RECORD ", color: .yellow))]"
let mutationString = "[\(setText("MUTATE ", color: .blue))]"

//------------------------------------------------------------------------------
// LogLevel
public enum LogLevel : Int, Comparable {
	case error, warning, status, diagnostic
	
	public init?(string: String) {
		switch string {
		case "error"     : self = .error
		case "warning"   : self = .warning
		case "status"    : self = .status
		case "diagnostic": self = .diagnostic
		default: return nil
		}
	}
}

public func <(lhs: LogLevel, rhs: LogLevel) -> Bool {
	return lhs.rawValue < rhs.rawValue
}



