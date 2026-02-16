#!/usr/bin/env python3
"""
Complete NSCK AI Model Evaluation Suite
========================================

This script runs the complete evaluation suite including:
1. Comprehensive benchmarking
2. Telemetry monitoring
3. Extended training and testing
4. Final analysis and reporting

Usage:
    python run_complete_evaluation.py [--output-dir DIR] [--extended]
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsck_ai_model.ai_engine import NSCKAIEngine
from nsck_ai_model.comprehensive_benchmark import ComprehensiveBenchmark
from nsck_ai_model.telemetry_monitor import TelemetryMonitor, MonitoredEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
)
logger = logging.getLogger("complete_evaluation")


def run_complete_evaluation(output_dir: str = "./complete_evaluation", extended: bool = False):
    """Run the complete evaluation suite."""
    
    logger.info("=" * 80)
    logger.info("NSCK AI MODEL - COMPLETE EVALUATION SUITE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Extended mode: {extended}")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Comprehensive Benchmark
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: COMPREHENSIVE BENCHMARK")
    logger.info("=" * 80)
    
    benchmark_dir = os.path.join(output_dir, "benchmark")
    benchmark = ComprehensiveBenchmark(output_dir=benchmark_dir)
    
    try:
        benchmark_report = benchmark.run_complete_benchmark()
        logger.info("✓ Comprehensive benchmark completed")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        benchmark_report = {"status": "failed", "error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Telemetry Monitoring with Extended Training
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: TELEMETRY MONITORING")
    logger.info("=" * 80)
    
    telemetry_dir = os.path.join(output_dir, "telemetry")
    monitor = TelemetryMonitor(output_dir=telemetry_dir)
    
    # Create monitored engine
    engine = NSCKAIEngine()
    monitored_engine = MonitoredEngine(engine, monitor)
    
    # Extended training if requested
    if extended:
        logger.info("Running extended training with telemetry...")
        _run_extended_training(monitored_engine, monitor)
    else:
        logger.info("Running standard training with telemetry...")
        _run_standard_training(monitored_engine, monitor)
    
    # Run monitored queries
    logger.info("Running monitored query tests...")
    _run_monitored_queries(monitored_engine, monitor)
    
    # Generate telemetry report
    telemetry_report = monitor.generate_monitoring_report()
    monitor.export_query_log()
    
    logger.info("✓ Telemetry monitoring completed")
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: Performance Stress Test
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: PERFORMANCE STRESS TEST")
    logger.info("=" * 80)
    
    stress_results = _run_stress_test(monitored_engine, monitor)
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Compile Final Report
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: COMPILING FINAL REPORT")
    logger.info("=" * 80)
    
    total_time = time.time() - start_time
    
    final_report = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "total_duration_s": total_time,
            "extended_mode": extended,
            "output_directory": output_dir,
        },
        "benchmark_summary": _extract_benchmark_summary(benchmark_report),
        "telemetry_summary": _extract_telemetry_summary(telemetry_report),
        "stress_test_results": stress_results,
        "overall_assessment": _generate_overall_assessment(
            benchmark_report, telemetry_report, stress_results
        ),
    }
    
    # Save final report
    report_file = os.path.join(output_dir, "COMPLETE_EVALUATION_REPORT.json")
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    # Generate markdown report
    _generate_markdown_report(final_report, output_dir)
    
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total duration: {total_time:.2f}s")
    logger.info(f"Results saved to: {output_dir}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n📊 Benchmark Pass Rate: {final_report['benchmark_summary'].get('pass_rate', 'N/A')}")
    print(f"📈 Average Confidence: {final_report['telemetry_summary'].get('avg_confidence', 'N/A')}")
    print(f"⚡ Average Latency: {final_report['telemetry_summary'].get('avg_latency_ms', 'N/A')}ms")
    print(f"🔍 Anomalies Detected: {final_report['telemetry_summary'].get('anomaly_count', 'N/A')}")
    print(f"💪 Stress Test: {stress_results.get('status', 'N/A')}")
    print(f"\n📁 Full report: {report_file}")
    print("=" * 80)
    
    return final_report


def _run_standard_training(monitored_engine, monitor):
    """Run standard training regime."""
    monitor.start_training_session("standard_training")
    
    training_data = [
        # Science
        "Photosynthesis converts light energy into chemical energy in plants.",
        "The process of evolution occurs through natural selection over generations.",
        "Atoms are the basic building blocks of matter.",
        "Chemical reactions involve the breaking and forming of chemical bonds.",
        "The periodic table organizes elements by atomic number and properties.",
        
        # Mathematics
        "Prime numbers are only divisible by one and themselves.",
        "The Fibonacci sequence appears frequently in nature.",
        "Calculus enables the study of rates of change and accumulation.",
        
        # History
        "The Renaissance was a period of cultural rebirth in Europe.",
        "The printing press revolutionized the spread of information.",
        
        # Technology
        "Computers process information using binary logic.",
        "The internet connects billions of devices worldwide.",
        "Artificial intelligence enables machines to learn from data.",
    ]
    
    for text in training_data:
        monitored_engine.train_on_text(text)
    
    monitor.end_training_session()
    logger.info(f"  Trained on {len(training_data)} samples")


def _run_extended_training(monitored_engine, monitor):
    """Run extended training with more data."""
    monitor.start_training_session("extended_training")
    
    # Load extended training data
    extended_data = _get_extended_training_data()
    
    logger.info(f"  Training on {len(extended_data)} samples (this may take a while)...")
    
    for i, text in enumerate(extended_data):
        monitored_engine.train_on_text(text)
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(extended_data)} samples")
    
    monitor.end_training_session()
    logger.info(f"  ✓ Trained on {len(extended_data)} samples")


def _get_extended_training_data():
    """Get extended training dataset."""
    return [
        # Physics
        "Energy can neither be created nor destroyed, only transformed.",
        "Light exhibits both wave and particle properties.",
        "Quantum mechanics describes behavior at atomic scales.",
        "Relativity explains gravity as curvature of spacetime.",
        "Thermodynamics governs heat transfer and energy conversion.",
        
        # Biology
        "Cells are the fundamental units of life.",
        "DNA stores genetic information in all living organisms.",
        "Proteins perform most cellular functions.",
        "Enzymes catalyze biochemical reactions.",
        "Natural selection drives evolutionary change.",
        "Ecosystems consist of interacting organisms and environment.",
        
        # Chemistry
        "Elements combine to form compounds through chemical bonds.",
        "Acids donate protons while bases accept them.",
        "Catalysts speed up reactions without being consumed.",
        "Oxidation involves loss of electrons, reduction involves gain.",
        
        # Computer Science
        "Algorithms are step-by-step procedures for solving problems.",
        "Data structures organize information for efficient access.",
        "Operating systems manage computer hardware and software resources.",
        "Networks enable communication between computers.",
        "Databases store and retrieve structured information.",
        "Encryption protects data through mathematical transformations.",
        
        # Psychology
        "Memory involves encoding, storage, and retrieval of information.",
        "Cognitive biases systematically affect judgment and decision-making.",
        "Learning occurs through association, reinforcement, and observation.",
        
        # Philosophy
        "Ethics examines principles of right and wrong behavior.",
        "Epistemology studies the nature and limits of knowledge.",
        "Logic provides rules for valid reasoning.",
        
        # Economics
        "Supply and demand determine prices in market economies.",
        "Opportunity cost represents the value of the next best alternative.",
        "Comparative advantage drives international trade.",
        
        # More scientific facts
        "The speed of light in vacuum is approximately 299,792 kilometers per second.",
        "Gravity decreases with the square of distance.",
        "Chemical equilibrium occurs when forward and reverse reaction rates are equal.",
        "Entropy measures disorder in a system.",
        "Neurons communicate through electrical and chemical signals.",
        "DNA replication is semi-conservative.",
        "Mitosis produces two identical daughter cells.",
        "Meiosis produces four genetically diverse gametes.",
    ]


def _run_monitored_queries(monitored_engine, monitor):
    """Run a series of monitored queries."""
    
    test_queries = [
        # Factual
        "What is photosynthesis?",
        "Explain natural selection.",
        "What is a prime number?",
        
        # Reasoning
        "How are DNA and proteins related?",
        "Why is the speed of light important?",
        "What connects computers and networks?",
        
        # Complex
        "Explain the relationship between energy and matter.",
        "How do cells store and use genetic information?",
        
        # Abstract
        "What is the nature of knowledge?",
        "How does learning occur?",
    ]
    
    for query in test_queries:
        result = monitored_engine.chat(query)
        logger.info(f"  Q: {query[:50]}... | Confidence: {result['confidence']:.2f}")


def _run_stress_test(monitored_engine, monitor):
    """Run stress test to check system limits."""
    logger.info("Running stress test...")
    
    # Burst test
    burst_size = 100
    burst_start = time.time()
    
    for i in range(burst_size):
        monitored_engine.chat("What is AI?")
    
    burst_duration = time.time() - burst_start
    burst_qps = burst_size / burst_duration
    
    logger.info(f"  Burst test: {burst_size} queries in {burst_duration:.2f}s = {burst_qps:.1f} QPS")
    
    # Long conversation test
    conversation_length = 20
    conversation_start = time.time()
    
    for i in range(conversation_length):
        monitored_engine.chat(f"Tell me fact number {i + 1}.")
    
    conversation_duration = time.time() - conversation_start
    
    logger.info(f"  Conversation test: {conversation_length} turns in {conversation_duration:.2f}s")
    
    # Get system stats
    stats = monitored_engine.get_system_stats()
    
    return {
        "status": "completed",
        "burst_qps": burst_qps,
        "burst_duration_s": burst_duration,
        "conversation_duration_s": conversation_duration,
        "final_concepts": stats['knowledge']['total_concepts'],
        "final_relations": stats['knowledge']['total_relations'],
    }


def _extract_benchmark_summary(benchmark_report):
    """Extract summary from benchmark report."""
    if isinstance(benchmark_report, dict) and benchmark_report.get('status') == 'failed':
        return {"status": "failed", "error": benchmark_report.get('error')}
    
    try:
        return {
            "datasets_trained": benchmark_report.get('training_summary', {}).get('datasets_trained', 0),
            "concepts_learned": benchmark_report.get('training_summary', {}).get('total_concepts_learned', 0),
            "cognitive_tests_passed": benchmark_report.get('cognitive_summary', {}).get('tests_passed', 0),
            "pass_rate": f"{benchmark_report.get('cognitive_summary', {}).get('pass_rate', 0):.1%}",
            "strengths_count": len(benchmark_report.get('strengths', [])),
            "weaknesses_count": len(benchmark_report.get('weaknesses', [])),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _extract_telemetry_summary(telemetry_report):
    """Extract summary from telemetry report."""
    try:
        stats = telemetry_report.get('overall_stats', {})
        return {
            "total_queries": stats.get('total_queries', 0),
            "training_samples": stats.get('total_training_samples', 0),
            "avg_confidence": f"{stats.get('avg_confidence', 0):.2%}",
            "avg_latency_ms": f"{stats.get('avg_latency_ms', 0):.2f}",
            "anomaly_count": stats.get('anomaly_count', 0),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _generate_overall_assessment(benchmark_report, telemetry_report, stress_results):
    """Generate overall assessment."""
    
    assessment = []
    
    # Performance assessment
    try:
        qps = stress_results.get('burst_qps', 0)
        if qps > 100:
            assessment.append("Excellent throughput performance (>100 QPS)")
        elif qps > 50:
            assessment.append("Good throughput performance (50-100 QPS)")
        else:
            assessment.append("Moderate throughput performance (<50 QPS)")
    except:
        pass
    
    # Cognitive assessment
    try:
        pass_rate = benchmark_report.get('cognitive_summary', {}).get('pass_rate', 0)
        if pass_rate >= 0.8:
            assessment.append("Strong cognitive capabilities (≥80% pass rate)")
        elif pass_rate >= 0.6:
            assessment.append("Good cognitive capabilities (60-80% pass rate)")
        else:
            assessment.append("Developing cognitive capabilities (<60% pass rate)")
    except:
        pass
    
    # Reliability assessment
    try:
        anomaly_count = telemetry_report.get('overall_stats', {}).get('anomaly_count', 0)
        total_queries = telemetry_report.get('overall_stats', {}).get('total_queries', 1)
        anomaly_rate = anomaly_count / total_queries if total_queries > 0 else 0
        
        if anomaly_rate < 0.05:
            assessment.append("High reliability (<5% anomaly rate)")
        elif anomaly_rate < 0.15:
            assessment.append("Good reliability (5-15% anomaly rate)")
        else:
            assessment.append("Needs reliability improvements (>15% anomaly rate)")
    except:
        pass
    
    return assessment


def _generate_markdown_report(report, output_dir):
    """Generate markdown version of the report."""
    filepath = os.path.join(output_dir, "COMPLETE_EVALUATION_REPORT.md")
    
    with open(filepath, 'w') as f:
        f.write("# NSCK AI Model - Complete Evaluation Report\n\n")
        
        # Metadata
        meta = report['metadata']
        f.write("## Evaluation Metadata\n\n")
        f.write(f"- **Date**: {meta['evaluation_date']}\n")
        f.write(f"- **Duration**: {meta['total_duration_s']:.2f}s\n")
        f.write(f"- **Mode**: {'Extended' if meta['extended_mode'] else 'Standard'}\n\n")
        
        # Benchmark Summary
        f.write("## Benchmark Summary\n\n")
        bench = report['benchmark_summary']
        for key, value in bench.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n")
        
        # Telemetry Summary
        f.write("## Telemetry Summary\n\n")
        telem = report['telemetry_summary']
        for key, value in telem.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n")
        
        # Stress Test
        f.write("## Stress Test Results\n\n")
        stress = report['stress_test_results']
        for key, value in stress.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n")
        
        # Overall Assessment
        f.write("## Overall Assessment\n\n")
        for item in report['overall_assessment']:
            f.write(f"- {item}\n")
        f.write("\n")
    
    logger.info(f"Saved markdown report to {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Complete NSCK AI Model Evaluation Suite"
    )
    parser.add_argument(
        "--output-dir",
        default="./complete_evaluation",
        help="Output directory for all results"
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run extended evaluation with more training data"
    )
    
    args = parser.parse_args()
    
    try:
        run_complete_evaluation(
            output_dir=args.output_dir,
            extended=args.extended
        )
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
