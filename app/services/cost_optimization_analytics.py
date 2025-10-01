"""
Cost and resource optimization analytics service.
Task 5.3: Create cost and resource optimization analytics
"""

import asyncio
import time
import statistics
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict, Counter
import json
import math


class CostCategory(Enum):
    """Categories of crawling costs."""
    COMPUTE = "compute"
    BANDWIDTH = "bandwidth"
    STORAGE = "storage"
    BROWSER_INSTANCE = "browser_instance"
    RETRY_OVERHEAD = "retry_overhead"
    INFRASTRUCTURE = "infrastructure"


class OptimizationType(Enum):
    """Types of optimization recommendations."""
    COST_REDUCTION = "cost_reduction"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SCALING_OPTIMIZATION = "scaling_optimization"
    SCHEDULE_OPTIMIZATION = "schedule_optimization"
    RETRY_OPTIMIZATION = "retry_optimization"
    CAPACITY_PLANNING = "capacity_planning"


class AlertSeverity(Enum):
    """Severity levels for budget alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CrawlCost:
    """Cost information for a single crawl job."""
    job_id: str
    url: str
    start_time: datetime
    end_time: datetime
    success: bool
    pages_crawled: int
    bytes_downloaded: Optional[int] = None
    cpu_seconds: float = 0.0
    memory_mb_hours: float = 0.0
    bandwidth_gb: float = 0.0
    browser_instance_hours: float = 0.0
    storage_gb: float = 0.0
    retry_count: int = 0
    concurrent_jobs: Optional[int] = None
    user_id: Optional[str] = None
    site_category: Optional[str] = None


@dataclass
class ResourceUsage:
    """Resource usage metrics for analysis."""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    bandwidth_utilization: float
    storage_utilization: float
    active_browser_instances: int
    concurrent_jobs: int
    queue_depth: int


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation with impact analysis."""
    recommendation_id: str
    recommendation_type: OptimizationType
    title: str
    description: str
    impact_analysis: Dict[str, Any]
    implementation_effort: str  # low, medium, high
    estimated_savings_monthly: float
    confidence_score: float
    priority: str  # low, medium, high, critical
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CapacityPlan:
    """Capacity planning recommendations."""
    current_capacity: Dict[str, float]
    projected_demand: Dict[str, float]
    recommended_capacity: Dict[str, float]
    scaling_milestones: List[Dict[str, Any]]
    cost_projections: Dict[str, float]
    timeline: Dict[str, datetime]


@dataclass
class BudgetAlert:
    """Budget-related alert."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    current_value: float
    threshold_value: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False


@dataclass
class ResourceEfficiencyMetrics:
    """Resource efficiency metrics and analysis."""
    pages_per_cpu_second: float
    pages_per_mb_hour: float
    cost_per_page: float
    bandwidth_efficiency: float
    time_efficiency: float
    resource_utilization_score: float
    efficiency_trend: str  # improving, declining, stable


class CostOptimizationAnalytics:
    """
    Comprehensive cost and resource optimization analytics service.

    Features:
    - Crawling cost analysis per job and site
    - Resource efficiency metrics and optimization
    - Capacity planning and scaling recommendations
    - Budget management and alerting
    - Cost forecasting and anomaly detection
    """

    def __init__(self,
                 monthly_budget: Optional[float] = None,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 pricing_config: Optional[Dict[str, float]] = None):
        """Initialize cost optimization analytics service."""
        self.monthly_budget = monthly_budget
        self.alert_thresholds = alert_thresholds or {
            'daily_budget_percent': 10.0,  # 10% of monthly budget per day
            'weekly_budget_percent': 25.0,  # 25% of monthly budget per week
            'monthly_budget_percent': 90.0  # Alert at 90% of monthly budget
        }

        # Default pricing configuration (per unit costs)
        self.pricing_config = pricing_config or {
            'cpu_second': 0.0001,  # $0.0001 per CPU second
            'memory_mb_hour': 0.000015,  # $0.000015 per MB-hour
            'bandwidth_gb': 0.10,  # $0.10 per GB bandwidth
            'browser_instance_hour': 0.50,  # $0.50 per browser instance hour
            'storage_gb': 0.05,  # $0.05 per GB storage per month
            'retry_penalty': 0.5  # 50% additional cost per retry
        }

        self.crawl_costs: List[CrawlCost] = []
        self.resource_usage_history: List[ResourceUsage] = []
        self.budget_alerts: List[BudgetAlert] = []
        self.optimization_recommendations: List[OptimizationRecommendation] = []

        # Caching
        self._cost_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize the cost optimization analytics service."""
        await self._cleanup_old_data()

    async def shutdown(self):
        """Shutdown the cost optimization analytics service."""
        # Perform any cleanup if needed
        pass

    def record_crawl_cost(self, crawl_cost: CrawlCost):
        """Record cost information for a crawl job."""
        self.crawl_costs.append(crawl_cost)

        # Real-time cost monitoring
        self._check_real_time_budget_alerts(crawl_cost)

        # Clear relevant caches
        self._invalidate_cache(['site_costs', 'cost_breakdown', 'efficiency_metrics'])

    def record_resource_usage(self, resource_usage: ResourceUsage):
        """Record resource usage metrics."""
        self.resource_usage_history.append(resource_usage)

        # Clear relevant caches
        self._invalidate_cache(['resource_metrics', 'capacity_analysis'])

    def calculate_job_cost(self, job_id: str) -> Dict[str, float]:
        """Calculate total cost for a specific job."""
        crawl_cost = next((c for c in self.crawl_costs if c.job_id == job_id), None)
        if not crawl_cost:
            return {"error": "Job not found"}

        # Calculate individual cost components
        compute_cost = (
            crawl_cost.cpu_seconds * self.pricing_config['cpu_second'] +
            crawl_cost.memory_mb_hours * self.pricing_config['memory_mb_hour']
        )

        bandwidth_cost = crawl_cost.bandwidth_gb * self.pricing_config['bandwidth_gb']

        storage_cost = crawl_cost.storage_gb * self.pricing_config['storage_gb']

        browser_cost = crawl_cost.browser_instance_hours * self.pricing_config['browser_instance_hour']

        # Retry penalty
        retry_cost = 0.0
        if crawl_cost.retry_count > 0:
            base_cost = compute_cost + bandwidth_cost + browser_cost
            retry_cost = base_cost * crawl_cost.retry_count * self.pricing_config['retry_penalty']

        total_cost = compute_cost + bandwidth_cost + storage_cost + browser_cost + retry_cost

        return {
            "job_id": job_id,
            "total_cost": total_cost,
            "compute_cost": compute_cost,
            "bandwidth_cost": bandwidth_cost,
            "storage_cost": storage_cost,
            "browser_instance_cost": browser_cost,
            "retry_cost": retry_cost,
            "cost_per_page": total_cost / max(crawl_cost.pages_crawled, 1)
        }

    def analyze_site_costs(self, time_window: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        """Analyze costs per site."""
        cache_key = f"site_costs_{time_window}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        filtered_costs = self._filter_crawl_costs(time_window)
        site_costs = defaultdict(list)

        # Group costs by site
        for crawl_cost in filtered_costs:
            site_url = self._extract_domain(crawl_cost.url)
            job_cost = self.calculate_job_cost(crawl_cost.job_id)
            if "error" not in job_cost:
                site_costs[site_url].append({
                    'cost': job_cost['total_cost'],
                    'pages': crawl_cost.pages_crawled,
                    'success': crawl_cost.success,
                    'retries': crawl_cost.retry_count
                })

        # Calculate site-level metrics
        result = []
        for site_url, costs in site_costs.items():
            total_cost = sum(c['cost'] for c in costs)
            total_pages = sum(c['pages'] for c in costs)
            total_jobs = len(costs)
            successful_jobs = sum(1 for c in costs if c['success'])
            total_retries = sum(c['retries'] for c in costs)

            result.append({
                'site_url': site_url,
                'total_cost': total_cost,
                'total_jobs': total_jobs,
                'successful_jobs': successful_jobs,
                'success_rate': (successful_jobs / total_jobs) * 100 if total_jobs > 0 else 0,
                'total_pages': total_pages,
                'average_cost_per_crawl': total_cost / total_jobs if total_jobs > 0 else 0,
                'cost_per_page': total_cost / total_pages if total_pages > 0 else 0,
                'retry_rate': total_retries / total_jobs if total_jobs > 0 else 0,
                'efficiency_score': self._calculate_site_efficiency_score(costs)
            })

        # Sort by total cost descending
        result.sort(key=lambda x: x['total_cost'], reverse=True)

        self._cache_result(cache_key, result)
        return result

    def get_cost_breakdown_by_category(self,
                                     time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get cost breakdown by category."""
        cache_key = f"cost_breakdown_{time_window}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        filtered_costs = self._filter_crawl_costs(time_window)
        category_costs = defaultdict(float)

        for crawl_cost in filtered_costs:
            job_cost = self.calculate_job_cost(crawl_cost.job_id)
            if "error" not in job_cost:
                category_costs[CostCategory.COMPUTE.value] += job_cost['compute_cost']
                category_costs[CostCategory.BANDWIDTH.value] += job_cost['bandwidth_cost']
                category_costs[CostCategory.STORAGE.value] += job_cost['storage_cost']
                category_costs[CostCategory.BROWSER_INSTANCE.value] += job_cost['browser_instance_cost']
                category_costs[CostCategory.RETRY_OVERHEAD.value] += job_cost['retry_cost']

        result = dict(category_costs)
        self._cache_result(cache_key, result)
        return result

    def calculate_resource_efficiency(self) -> ResourceEfficiencyMetrics:
        """Calculate resource efficiency metrics."""
        cache_key = "efficiency_metrics"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        if not self.crawl_costs:
            return ResourceEfficiencyMetrics(
                pages_per_cpu_second=0.0, pages_per_mb_hour=0.0,
                cost_per_page=0.0, bandwidth_efficiency=0.0,
                time_efficiency=0.0, resource_utilization_score=0.0,
                efficiency_trend="stable"
            )

        # Calculate efficiency metrics
        total_pages = sum(c.pages_crawled for c in self.crawl_costs)
        total_cpu_seconds = sum(c.cpu_seconds for c in self.crawl_costs)
        total_memory_mb_hours = sum(c.memory_mb_hours for c in self.crawl_costs)
        total_bandwidth_gb = sum(c.bandwidth_gb for c in self.crawl_costs)
        total_time_hours = sum((c.end_time - c.start_time).total_seconds() / 3600 for c in self.crawl_costs)

        pages_per_cpu_second = total_pages / total_cpu_seconds if total_cpu_seconds > 0 else 0
        pages_per_mb_hour = total_pages / total_memory_mb_hours if total_memory_mb_hours > 0 else 0

        # Calculate total cost and cost per page
        total_cost = sum(
            self.calculate_job_cost(c.job_id).get('total_cost', 0)
            for c in self.crawl_costs
        )
        cost_per_page = total_cost / total_pages if total_pages > 0 else 0

        # Bandwidth efficiency (pages per GB)
        bandwidth_efficiency = total_pages / total_bandwidth_gb if total_bandwidth_gb > 0 else 0

        # Time efficiency (pages per hour)
        time_efficiency = total_pages / total_time_hours if total_time_hours > 0 else 0

        # Overall resource utilization score (0-100)
        resource_utilization_score = self._calculate_resource_utilization_score()

        # Efficiency trend
        efficiency_trend = self._calculate_efficiency_trend()

        result = ResourceEfficiencyMetrics(
            pages_per_cpu_second=pages_per_cpu_second,
            pages_per_mb_hour=pages_per_mb_hour,
            cost_per_page=cost_per_page,
            bandwidth_efficiency=bandwidth_efficiency,
            time_efficiency=time_efficiency,
            resource_utilization_score=resource_utilization_score,
            efficiency_trend=efficiency_trend
        )

        self._cache_result(cache_key, result)
        return result

    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate cost and resource optimization recommendations."""
        recommendations = []

        # Analyze retry patterns
        retry_rec = self._analyze_retry_optimization()
        if retry_rec:
            recommendations.append(retry_rec)

        # Analyze resource inefficiencies
        resource_recs = self._analyze_resource_inefficiencies()
        recommendations.extend(resource_recs)

        # Analyze scheduling optimization opportunities
        schedule_rec = self._analyze_schedule_optimization()
        if schedule_rec:
            recommendations.append(schedule_rec)

        # Analyze capacity optimization
        capacity_recs = self._analyze_capacity_optimization()
        recommendations.extend(capacity_recs)

        # Sort by potential savings
        recommendations.sort(
            key=lambda x: x.estimated_savings_monthly,
            reverse=True
        )

        return recommendations

    def generate_scaling_recommendations(self) -> Dict[str, Any]:
        """Generate scaling recommendations based on usage patterns."""
        if not self.resource_usage_history:
            return {
                "current_utilization": {},
                "recommended_capacity": {},
                "scaling_strategy": "insufficient_data",
                "cost_impact": {}
            }

        # Calculate current utilization averages
        recent_usage = self.resource_usage_history[-24:]  # Last 24 data points
        if not recent_usage:
            recent_usage = self.resource_usage_history

        avg_cpu = statistics.mean(u.cpu_utilization for u in recent_usage)
        avg_memory = statistics.mean(u.memory_utilization for u in recent_usage)
        avg_bandwidth = statistics.mean(u.bandwidth_utilization for u in recent_usage)
        max_concurrent = max(u.concurrent_jobs for u in recent_usage) if recent_usage else 0

        current_utilization = {
            "cpu_percent": avg_cpu,
            "memory_percent": avg_memory,
            "bandwidth_percent": avg_bandwidth,
            "max_concurrent_jobs": max_concurrent
        }

        # Determine scaling strategy
        scaling_strategy = "maintain"
        recommended_capacity = current_utilization.copy()

        if avg_cpu > 80 or avg_memory > 80:
            scaling_strategy = "scale_up"
            recommended_capacity["cpu_percent"] = avg_cpu * 1.5
            recommended_capacity["memory_percent"] = avg_memory * 1.5
        elif avg_cpu < 30 and avg_memory < 30:
            scaling_strategy = "scale_down"
            recommended_capacity["cpu_percent"] = avg_cpu * 0.7
            recommended_capacity["memory_percent"] = avg_memory * 0.7

        # Estimate cost impact
        current_cost = self._estimate_infrastructure_cost(current_utilization)
        projected_cost = self._estimate_infrastructure_cost(recommended_capacity)
        cost_impact = {
            "current_monthly_cost": current_cost,
            "projected_monthly_cost": projected_cost,
            "cost_change": projected_cost - current_cost,
            "cost_change_percent": ((projected_cost - current_cost) / current_cost * 100) if current_cost > 0 else 0
        }

        return {
            "current_utilization": current_utilization,
            "recommended_capacity": recommended_capacity,
            "scaling_strategy": scaling_strategy,
            "cost_impact": cost_impact
        }

    def generate_budget_optimization(self) -> Dict[str, Any]:
        """Generate budget optimization recommendations."""
        if not self.crawl_costs:
            return {"error": "No cost data available"}

        # Calculate current spending patterns
        monthly_costs = self._calculate_monthly_costs()
        current_monthly_cost = monthly_costs.get('current_month', 0)

        # Project future costs based on trends
        projected_cost = self._project_monthly_cost()

        # Identify optimization opportunities
        optimization_opportunities = []

        # High retry sites
        high_retry_sites = [
            site for site in self.analyze_site_costs()
            if site['retry_rate'] > 2.0  # More than 2 retries per job on average
        ]
        if high_retry_sites:
            potential_savings = sum(site['total_cost'] * 0.3 for site in high_retry_sites)
            optimization_opportunities.append({
                'type': 'retry_optimization',
                'description': f'Optimize {len(high_retry_sites)} sites with high retry rates',
                'potential_monthly_savings': potential_savings,
                'implementation_effort': 'medium'
            })

        # Expensive sites per page
        expensive_sites = [
            site for site in self.analyze_site_costs()
            if site['cost_per_page'] > 0.01  # More than $0.01 per page
        ]
        if expensive_sites:
            potential_savings = sum(site['total_cost'] * 0.2 for site in expensive_sites)
            optimization_opportunities.append({
                'type': 'efficiency_optimization',
                'description': f'Optimize {len(expensive_sites)} sites with high cost per page',
                'potential_monthly_savings': potential_savings,
                'implementation_effort': 'high'
            })

        total_savings_potential = sum(opp['potential_monthly_savings'] for opp in optimization_opportunities)

        return {
            "current_monthly_cost": current_monthly_cost,
            "projected_cost": projected_cost,
            "monthly_budget": self.monthly_budget,
            "budget_utilization": (current_monthly_cost / self.monthly_budget * 100) if self.monthly_budget else None,
            "optimization_opportunities": optimization_opportunities,
            "savings_potential": total_savings_potential,
            "optimized_projected_cost": projected_cost - total_savings_potential
        }

    def generate_capacity_plan(self, forecast_days: int = 30) -> CapacityPlan:
        """Generate capacity planning recommendations."""
        if not self.crawl_costs or not self.resource_usage_history:
            return CapacityPlan(
                current_capacity={}, projected_demand={},
                recommended_capacity={}, scaling_milestones=[],
                cost_projections={}, timeline={}
            )

        # Analyze current capacity
        current_capacity = self._analyze_current_capacity()

        # Project demand growth
        projected_demand = self._project_demand_growth(forecast_days)

        # Calculate recommended capacity
        recommended_capacity = {}
        for resource, demand in projected_demand.items():
            current = current_capacity.get(resource, 0)
            buffer = 0.2  # 20% buffer
            recommended_capacity[resource] = demand * (1 + buffer)

        # Create scaling milestones
        scaling_milestones = self._create_scaling_milestones(
            current_capacity, projected_demand, forecast_days
        )

        # Project costs
        cost_projections = {
            "current_monthly": self._estimate_infrastructure_cost(current_capacity),
            "projected_monthly": self._estimate_infrastructure_cost(recommended_capacity),
            "scaling_investment": self._estimate_scaling_investment(scaling_milestones)
        }

        # Create timeline
        timeline = {
            "assessment_date": datetime.now(timezone.utc),
            "next_review_date": datetime.now(timezone.utc) + timedelta(days=30),
            "scaling_deadline": datetime.now(timezone.utc) + timedelta(days=forecast_days)
        }

        return CapacityPlan(
            current_capacity=current_capacity,
            projected_demand=projected_demand,
            recommended_capacity=recommended_capacity,
            scaling_milestones=scaling_milestones,
            cost_projections=cost_projections,
            timeline=timeline
        )

    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation based on workload patterns."""
        if not self.crawl_costs:
            return {"error": "Insufficient data for optimization"}

        # Analyze workload patterns
        workload_analysis = self._analyze_workload_patterns()

        # Calculate current allocation efficiency
        current_efficiency = self._calculate_allocation_efficiency()

        # Generate optimized allocation
        optimized_allocation = self._generate_optimized_allocation(workload_analysis)

        # Calculate improvement potential
        efficiency_improvement = self._calculate_efficiency_improvement(
            current_efficiency, optimized_allocation
        )

        return {
            "current_allocation": current_efficiency,
            "optimized_allocation": optimized_allocation,
            "efficiency_improvement": efficiency_improvement,
            "cpu_allocation": optimized_allocation.get('cpu', {}),
            "memory_allocation": optimized_allocation.get('memory', {}),
            "bandwidth_allocation": optimized_allocation.get('bandwidth', {}),
            "cost_savings": efficiency_improvement.get('cost_savings', 0),
            "performance_improvement": efficiency_improvement.get('performance_improvement', 0),
            "recommendations": self._generate_allocation_recommendations(optimized_allocation)
        }

    def analyze_scaling_scenarios(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze different scaling scenarios and their cost implications."""
        results = []

        baseline_cost = self._calculate_baseline_monthly_cost()
        baseline_performance = self._calculate_baseline_performance()

        for scenario in scenarios:
            scale_factor = scenario.get('scale_factor', 1.0)
            scenario_name = scenario.get('name', f'scale_{scale_factor}x')

            # Project resource requirements
            resource_requirements = self._project_scenario_resources(scale_factor)

            # Calculate costs
            projected_cost = baseline_cost * scale_factor * 0.85  # Economies of scale

            # Estimate performance impact
            performance_impact = self._estimate_performance_impact(scale_factor)

            # ROI Analysis
            revenue_increase = scenario.get('expected_revenue_increase', scale_factor - 1)
            cost_increase = (projected_cost - baseline_cost) / baseline_cost
            roi = (revenue_increase - cost_increase) / cost_increase if cost_increase > 0 else 0

            results.append({
                "scenario_name": scenario_name,
                "scale_factor": scale_factor,
                "projected_cost": projected_cost,
                "cost_increase_percent": cost_increase * 100,
                "resource_requirements": resource_requirements,
                "performance_impact": performance_impact,
                "roi_analysis": {
                    "roi_percent": roi * 100,
                    "payback_months": 12 / roi if roi > 0 else float('inf'),
                    "break_even_scale": baseline_cost / (baseline_cost - projected_cost) if projected_cost < baseline_cost else None
                }
            })

        return results

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status and spending analysis."""
        if not self.monthly_budget:
            return {"error": "No budget configured"}

        # Calculate current spending
        current_month_spending = self._calculate_current_month_spending()
        remaining_budget = self.monthly_budget - current_month_spending

        # Project monthly spending
        days_elapsed = datetime.now(timezone.utc).day
        days_in_month = 30  # Simplified
        projected_monthly = (current_month_spending / days_elapsed) * days_in_month

        # Calculate utilization
        budget_utilization = (current_month_spending / self.monthly_budget) * 100

        return {
            "monthly_budget": self.monthly_budget,
            "current_spending": current_month_spending,
            "remaining_budget": remaining_budget,
            "projected_monthly_spending": projected_monthly,
            "budget_utilization_percent": budget_utilization,
            "days_elapsed": days_elapsed,
            "burn_rate": current_month_spending / days_elapsed if days_elapsed > 0 else 0,
            "budget_status": self._determine_budget_status(budget_utilization)
        }

    def get_budget_alerts(self) -> List[BudgetAlert]:
        """Get active budget alerts."""
        return [alert for alert in self.budget_alerts if not alert.resolved]

    def generate_cost_forecast(self, forecast_days: int = 30) -> Dict[str, Any]:
        """Generate cost forecast based on historical data."""
        if len(self.crawl_costs) < 7:  # Need at least a week of data
            return {"error": "Insufficient historical data for forecasting"}

        # Calculate daily costs for historical period
        daily_costs = self._calculate_daily_costs()

        # Fit trend line
        trend_slope = self._calculate_cost_trend(daily_costs)

        # Generate daily projections
        base_cost = statistics.mean(list(daily_costs.values())[-7:])  # Last week average
        daily_projections = []

        for day in range(forecast_days):
            projected_cost = base_cost + (trend_slope * day)
            projected_cost = max(0, projected_cost)  # Ensure non-negative

            date = datetime.now(timezone.utc) + timedelta(days=day)
            daily_projections.append({
                "date": date.date().isoformat(),
                "projected_cost": projected_cost
            })

        # Calculate aggregated projections
        total_projected = sum(dp["projected_cost"] for dp in daily_projections)
        weekly_projections = [
            sum(daily_projections[i:i+7][j]["projected_cost"] for j in range(min(7, len(daily_projections[i:i+7]))))
            for i in range(0, forecast_days, 7)
        ]

        # Confidence interval
        historical_variance = statistics.variance(daily_costs.values()) if len(daily_costs) > 1 else 0
        confidence_interval = {
            "lower_bound": total_projected * 0.85,
            "upper_bound": total_projected * 1.15,
            "confidence_level": 80
        }

        return {
            "forecast_period_days": forecast_days,
            "daily_projections": daily_projections,
            "weekly_projections": weekly_projections,
            "monthly_projection": total_projected,
            "confidence_interval": confidence_interval,
            "trend_analysis": {
                "slope": trend_slope,
                "direction": "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
            }
        }

    def detect_cost_anomalies(self) -> List[Dict[str, Any]]:
        """Detect cost anomalies in recent spending."""
        if len(self.crawl_costs) < 10:
            return []

        anomalies = []

        # Calculate baseline cost statistics
        job_costs = []
        for crawl_cost in self.crawl_costs:
            cost_data = self.calculate_job_cost(crawl_cost.job_id)
            if "error" not in cost_data:
                job_costs.append({
                    'job_id': crawl_cost.job_id,
                    'cost': cost_data['total_cost'],
                    'cost_per_page': cost_data['cost_per_page'],
                    'timestamp': crawl_cost.start_time
                })

        if len(job_costs) < 5:
            return anomalies

        # Calculate statistical thresholds
        costs = [jc['cost'] for jc in job_costs]
        cost_per_page = [jc['cost_per_page'] for jc in job_costs if jc['cost_per_page'] > 0]

        cost_mean = statistics.mean(costs)
        cost_stdev = statistics.stdev(costs) if len(costs) > 1 else 0

        cpp_mean = statistics.mean(cost_per_page) if cost_per_page else 0
        cpp_stdev = statistics.stdev(cost_per_page) if len(cost_per_page) > 1 else 0

        # Detect anomalies (values beyond 2 standard deviations)
        for job_cost in job_costs[-20:]:  # Check last 20 jobs
            cost = job_cost['cost']
            cpp = job_cost['cost_per_page']

            # High cost anomaly
            if cost > cost_mean + 2 * cost_stdev:
                anomalies.append({
                    'job_id': job_cost['job_id'],
                    'anomaly_type': 'high_cost',
                    'cost': cost,
                    'expected_range': (cost_mean - 2 * cost_stdev, cost_mean + 2 * cost_stdev),
                    'severity': 'high' if cost > cost_mean + 3 * cost_stdev else 'medium',
                    'timestamp': job_cost['timestamp']
                })

            # High cost per page anomaly
            if cpp > 0 and cpp > cpp_mean + 2 * cpp_stdev:
                anomalies.append({
                    'job_id': job_cost['job_id'],
                    'anomaly_type': 'high_cost_per_page',
                    'cost_per_page': cpp,
                    'expected_range': (cpp_mean - 2 * cpp_stdev, cpp_mean + 2 * cpp_stdev),
                    'severity': 'high' if cpp > cpp_mean + 3 * cpp_stdev else 'medium',
                    'timestamp': job_cost['timestamp']
                })

        return anomalies

    def calculate_optimization_impact(self) -> Dict[str, Any]:
        """Calculate the impact of implemented optimizations."""
        if len(self.crawl_costs) < 20:
            return {"error": "Insufficient data to calculate optimization impact"}

        # Split data into before/after optimization periods
        midpoint = len(self.crawl_costs) // 2
        before_costs = self.crawl_costs[:midpoint]
        after_costs = self.crawl_costs[midpoint:]

        # Calculate average costs
        before_avg_cost = self._calculate_average_job_cost(before_costs)
        after_avg_cost = self._calculate_average_job_cost(after_costs)

        # Calculate improvements
        cost_reduction = before_avg_cost - after_avg_cost
        cost_reduction_percent = (cost_reduction / before_avg_cost * 100) if before_avg_cost > 0 else 0

        # Calculate efficiency improvements
        before_efficiency = self._calculate_period_efficiency(before_costs)
        after_efficiency = self._calculate_period_efficiency(after_costs)

        efficiency_improvement = after_efficiency - before_efficiency

        # Project annual savings
        monthly_savings = cost_reduction * len(after_costs) * (30 / len(after_costs))  # Approximate
        annual_savings = monthly_savings * 12

        return {
            "cost_reduction_percent": cost_reduction_percent,
            "efficiency_improvement_percent": efficiency_improvement,
            "monthly_savings": monthly_savings,
            "annual_savings_projection": annual_savings,
            "before_period": {
                "average_cost": before_avg_cost,
                "efficiency_score": before_efficiency
            },
            "after_period": {
                "average_cost": after_avg_cost,
                "efficiency_score": after_efficiency
            }
        }

    # Private helper methods

    def _filter_crawl_costs(self, time_window: Optional[timedelta] = None) -> List[CrawlCost]:
        """Filter crawl costs by time window."""
        if not time_window:
            return self.crawl_costs

        cutoff_time = datetime.now(timezone.utc) - time_window
        return [c for c in self.crawl_costs if c.start_time >= cutoff_time]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if "://" in url:
            return url.split("://")[1].split("/")[0]
        return url.split("/")[0]

    def _calculate_site_efficiency_score(self, site_costs: List[Dict[str, Any]]) -> float:
        """Calculate efficiency score for a site (0-100)."""
        if not site_costs:
            return 0.0

        # Factors: success rate, cost per page, retry rate
        success_rate = sum(1 for c in site_costs if c['success']) / len(site_costs)
        avg_cost_per_page = statistics.mean(c['cost'] / max(c['pages'], 1) for c in site_costs)
        avg_retry_rate = statistics.mean(c['retries'] for c in site_costs)

        # Normalize and weight factors
        success_score = success_rate * 40  # 40 points for success rate
        cost_score = max(0, 30 - avg_cost_per_page * 3000)  # 30 points for low cost
        retry_score = max(0, 30 - avg_retry_rate * 10)  # 30 points for low retries

        return min(100, success_score + cost_score + retry_score)

    def _calculate_resource_utilization_score(self) -> float:
        """Calculate overall resource utilization score."""
        if not self.resource_usage_history:
            return 50.0  # Default middle score

        recent_usage = self.resource_usage_history[-24:]  # Last 24 data points
        if not recent_usage:
            return 50.0

        # Calculate average utilization
        avg_cpu = statistics.mean(u.cpu_utilization for u in recent_usage)
        avg_memory = statistics.mean(u.memory_utilization for u in recent_usage)
        avg_bandwidth = statistics.mean(u.bandwidth_utilization for u in recent_usage)

        # Optimal utilization is around 70-80%
        def utilization_score(util):
            if 70 <= util <= 80:
                return 100
            elif util < 70:
                return 50 + (util / 70) * 50
            else:
                return max(0, 100 - (util - 80) * 5)

        cpu_score = utilization_score(avg_cpu)
        memory_score = utilization_score(avg_memory)
        bandwidth_score = utilization_score(avg_bandwidth)

        return (cpu_score + memory_score + bandwidth_score) / 3

    def _calculate_efficiency_trend(self) -> str:
        """Calculate efficiency trend direction."""
        if len(self.crawl_costs) < 10:
            return "stable"

        # Split into two periods
        midpoint = len(self.crawl_costs) // 2
        early_period = self.crawl_costs[:midpoint]
        recent_period = self.crawl_costs[midpoint:]

        early_efficiency = self._calculate_period_efficiency(early_period)
        recent_efficiency = self._calculate_period_efficiency(recent_period)

        diff = recent_efficiency - early_efficiency

        if diff > 5:
            return "improving"
        elif diff < -5:
            return "declining"
        else:
            return "stable"

    def _calculate_period_efficiency(self, costs: List[CrawlCost]) -> float:
        """Calculate efficiency score for a period."""
        if not costs:
            return 0.0

        total_pages = sum(c.pages_crawled for c in costs)
        total_cost = sum(
            self.calculate_job_cost(c.job_id).get('total_cost', 0)
            for c in costs
        )

        # Pages per dollar (higher is better)
        return (total_pages / total_cost) * 1000 if total_cost > 0 else 0

    def _analyze_retry_optimization(self) -> Optional[OptimizationRecommendation]:
        """Analyze retry patterns for optimization opportunities."""
        high_retry_jobs = [c for c in self.crawl_costs if c.retry_count > 2]

        if len(high_retry_jobs) < 5:
            return None

        # Calculate potential savings
        total_retry_cost = sum(
            self.calculate_job_cost(c.job_id).get('retry_cost', 0)
            for c in high_retry_jobs
        )

        if total_retry_cost < 10:  # Less than $10 in retry costs
            return None

        monthly_savings = total_retry_cost * (30 / 7)  # Extrapolate to monthly

        return OptimizationRecommendation(
            recommendation_id=f"retry_opt_{int(time.time())}",
            recommendation_type=OptimizationType.RETRY_OPTIMIZATION,
            title="Optimize High-Retry Sites",
            description=f"Reduce retry overhead for {len(high_retry_jobs)} jobs with excessive retries",
            impact_analysis={
                "affected_jobs": len(high_retry_jobs),
                "current_retry_cost": total_retry_cost,
                "optimization_potential": "60% reduction in retry costs"
            },
            implementation_effort="medium",
            estimated_savings_monthly=monthly_savings * 0.6,
            confidence_score=0.85,
            priority="high" if monthly_savings > 100 else "medium"
        )

    def _analyze_resource_inefficiencies(self) -> List[OptimizationRecommendation]:
        """Analyze resource usage for inefficiencies."""
        recommendations = []

        # Memory inefficiency
        high_memory_jobs = [
            c for c in self.crawl_costs
            if c.memory_mb_hours > 100 and c.pages_crawled < 10
        ]

        if len(high_memory_jobs) > 5:
            monthly_savings = len(high_memory_jobs) * 2.0  # Estimate $2 savings per job
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"memory_opt_{int(time.time())}",
                recommendation_type=OptimizationType.RESOURCE_EFFICIENCY,
                title="Optimize Memory Usage",
                description="Reduce memory consumption for low-throughput crawling jobs",
                impact_analysis={
                    "affected_jobs": len(high_memory_jobs),
                    "optimization_potential": "40% memory reduction"
                },
                implementation_effort="high",
                estimated_savings_monthly=monthly_savings,
                confidence_score=0.75,
                priority="medium"
            ))

        return recommendations

    def _analyze_schedule_optimization(self) -> Optional[OptimizationRecommendation]:
        """Analyze scheduling patterns for optimization."""
        # Simplified: look for jobs clustered in expensive time periods
        peak_hour_jobs = [
            c for c in self.crawl_costs
            if 9 <= c.start_time.hour <= 17  # Business hours
        ]

        if len(peak_hour_jobs) > len(self.crawl_costs) * 0.8:  # 80% in peak hours
            monthly_savings = len(peak_hour_jobs) * 0.1  # Small per-job savings

            return OptimizationRecommendation(
                recommendation_id=f"schedule_opt_{int(time.time())}",
                recommendation_type=OptimizationType.SCHEDULE_OPTIMIZATION,
                title="Optimize Crawling Schedule",
                description="Shift non-urgent crawls to off-peak hours for cost savings",
                impact_analysis={
                    "peak_hour_jobs": len(peak_hour_jobs),
                    "optimization_potential": "15% cost reduction through scheduling"
                },
                implementation_effort="low",
                estimated_savings_monthly=monthly_savings,
                confidence_score=0.70,
                priority="low"
            )

        return None

    def _analyze_capacity_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze capacity utilization for optimization."""
        recommendations = []

        if not self.resource_usage_history:
            return recommendations

        # Check for over-provisioning
        recent_usage = self.resource_usage_history[-24:]
        if recent_usage:
            avg_cpu = statistics.mean(u.cpu_utilization for u in recent_usage)
            avg_memory = statistics.mean(u.memory_utilization for u in recent_usage)

            if avg_cpu < 30 and avg_memory < 30:  # Under-utilized
                monthly_savings = 200  # Estimated savings from downsizing

                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"capacity_opt_{int(time.time())}",
                    recommendation_type=OptimizationType.CAPACITY_PLANNING,
                    title="Optimize Capacity Allocation",
                    description="Reduce over-provisioned resources based on actual usage",
                    impact_analysis={
                        "cpu_utilization": avg_cpu,
                        "memory_utilization": avg_memory,
                        "optimization_potential": "30% capacity reduction"
                    },
                    implementation_effort="medium",
                    estimated_savings_monthly=monthly_savings,
                    confidence_score=0.80,
                    priority="high"
                ))

        return recommendations

    def _estimate_infrastructure_cost(self, capacity: Dict[str, float]) -> float:
        """Estimate monthly infrastructure cost for given capacity."""
        # Simplified cost model
        base_cost = 100  # Base infrastructure cost
        cpu_cost = capacity.get('cpu_percent', 50) * 2  # $2 per CPU percentage point
        memory_cost = capacity.get('memory_percent', 50) * 1.5  # $1.5 per memory percentage

        return base_cost + cpu_cost + memory_cost

    def _analyze_current_capacity(self) -> Dict[str, float]:
        """Analyze current capacity utilization."""
        if not self.resource_usage_history:
            return {
                "cpu_capacity": 100.0,
                "memory_capacity": 100.0,
                "bandwidth_capacity": 100.0,
                "concurrent_jobs_capacity": 50.0
            }

        recent_usage = self.resource_usage_history[-24:]
        max_cpu = max(u.cpu_utilization for u in recent_usage)
        max_memory = max(u.memory_utilization for u in recent_usage)
        max_bandwidth = max(u.bandwidth_utilization for u in recent_usage)
        max_concurrent = max(u.concurrent_jobs for u in recent_usage)

        return {
            "cpu_capacity": max_cpu,
            "memory_capacity": max_memory,
            "bandwidth_capacity": max_bandwidth,
            "concurrent_jobs_capacity": max_concurrent
        }

    def _project_demand_growth(self, forecast_days: int) -> Dict[str, float]:
        """Project demand growth over forecast period."""
        current_capacity = self._analyze_current_capacity()

        # Simple linear growth projection
        growth_rate = 0.02  # 2% per day growth
        growth_factor = 1 + (growth_rate * forecast_days)

        return {
            resource: current_value * growth_factor
            for resource, current_value in current_capacity.items()
        }

    def _create_scaling_milestones(self, current: Dict[str, float],
                                 projected: Dict[str, float],
                                 forecast_days: int) -> List[Dict[str, Any]]:
        """Create scaling milestones."""
        milestones = []

        for resource, current_value in current.items():
            projected_value = projected.get(resource, current_value)

            if projected_value > current_value * 1.5:  # Significant increase needed
                milestone_date = datetime.now(timezone.utc) + timedelta(days=forecast_days//2)
                milestones.append({
                    "resource": resource,
                    "milestone_date": milestone_date,
                    "current_capacity": current_value,
                    "target_capacity": projected_value,
                    "scaling_factor": projected_value / current_value,
                    "urgency": "high" if projected_value > current_value * 2 else "medium"
                })

        return milestones

    def _estimate_scaling_investment(self, milestones: List[Dict[str, Any]]) -> float:
        """Estimate investment required for scaling."""
        total_investment = 0

        for milestone in milestones:
            scaling_factor = milestone.get('scaling_factor', 1.0)
            base_cost = 1000  # Base scaling cost
            total_investment += base_cost * (scaling_factor - 1)

        return total_investment

    def _analyze_workload_patterns(self) -> Dict[str, Any]:
        """Analyze workload patterns for optimization."""
        if not self.crawl_costs:
            return {}

        # Analyze by hour of day
        hourly_patterns = defaultdict(list)
        for crawl_cost in self.crawl_costs:
            hour = crawl_cost.start_time.hour
            hourly_patterns[hour].append(crawl_cost)

        # Find peak hours
        peak_hours = sorted(hourly_patterns.keys(), key=lambda h: len(hourly_patterns[h]), reverse=True)[:3]

        return {
            "peak_hours": peak_hours,
            "total_jobs": len(self.crawl_costs),
            "jobs_per_hour": {hour: len(jobs) for hour, jobs in hourly_patterns.items()},
            "peak_utilization": len(hourly_patterns[peak_hours[0]]) if peak_hours else 0
        }

    def _calculate_allocation_efficiency(self) -> Dict[str, Any]:
        """Calculate current allocation efficiency."""
        if not self.crawl_costs:
            return {}

        total_cpu = sum(c.cpu_seconds for c in self.crawl_costs)
        total_memory = sum(c.memory_mb_hours for c in self.crawl_costs)
        total_pages = sum(c.pages_crawled for c in self.crawl_costs)

        return {
            "cpu_efficiency": total_pages / total_cpu if total_cpu > 0 else 0,
            "memory_efficiency": total_pages / total_memory if total_memory > 0 else 0,
            "overall_efficiency": total_pages / (total_cpu + total_memory) if (total_cpu + total_memory) > 0 else 0
        }

    def _generate_optimized_allocation(self, workload_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized resource allocation."""
        # Simplified optimization
        peak_hours = workload_analysis.get('peak_hours', [])

        return {
            "cpu": {
                "peak_allocation": 80,  # 80% during peak hours
                "off_peak_allocation": 40,  # 40% during off-peak
                "peak_hours": peak_hours
            },
            "memory": {
                "peak_allocation": 75,
                "off_peak_allocation": 35,
                "peak_hours": peak_hours
            },
            "bandwidth": {
                "peak_allocation": 70,
                "off_peak_allocation": 30,
                "peak_hours": peak_hours
            }
        }

    def _calculate_efficiency_improvement(self, current: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate efficiency improvement from optimization."""
        current_efficiency = current.get('overall_efficiency', 1.0)

        # Estimate improvement (simplified)
        estimated_improvement = 0.25  # 25% improvement

        return {
            "efficiency_improvement": estimated_improvement * 100,
            "cost_savings": current_efficiency * estimated_improvement * 100,  # Estimated monthly savings
            "performance_improvement": estimated_improvement * 100
        }

    def _generate_allocation_recommendations(self, optimized_allocation: Dict[str, Any]) -> List[str]:
        """Generate allocation recommendations."""
        recommendations = []

        cpu_allocation = optimized_allocation.get('cpu', {})
        if cpu_allocation.get('peak_allocation', 0) > 70:
            recommendations.append("Scale CPU resources during peak hours")

        memory_allocation = optimized_allocation.get('memory', {})
        if memory_allocation.get('off_peak_allocation', 0) < 50:
            recommendations.append("Reduce memory allocation during off-peak hours")

        return recommendations

    def _calculate_baseline_monthly_cost(self) -> float:
        """Calculate baseline monthly cost."""
        if not self.crawl_costs:
            return 0.0

        total_cost = sum(
            self.calculate_job_cost(c.job_id).get('total_cost', 0)
            for c in self.crawl_costs
        )

        # Extrapolate to monthly based on data period
        days_of_data = 7  # Assume we have a week of data
        return total_cost * (30 / days_of_data)

    def _calculate_baseline_performance(self) -> Dict[str, float]:
        """Calculate baseline performance metrics."""
        if not self.crawl_costs:
            return {}

        total_pages = sum(c.pages_crawled for c in self.crawl_costs)
        total_time = sum((c.end_time - c.start_time).total_seconds() for c in self.crawl_costs)
        success_rate = sum(1 for c in self.crawl_costs if c.success) / len(self.crawl_costs)

        return {
            "pages_per_hour": (total_pages / total_time) * 3600 if total_time > 0 else 0,
            "success_rate": success_rate * 100,
            "average_response_time": total_time / len(self.crawl_costs) if self.crawl_costs else 0
        }

    def _project_scenario_resources(self, scale_factor: float) -> Dict[str, float]:
        """Project resource requirements for scaling scenario."""
        baseline_resources = {
            "cpu_cores": 4,
            "memory_gb": 16,
            "bandwidth_gbps": 1,
            "storage_tb": 0.5
        }

        # Apply economies of scale (efficiency improves with scale)
        efficiency_factor = 1 - (scale_factor - 1) * 0.1  # 10% efficiency gain per scale unit

        return {
            resource: value * scale_factor * efficiency_factor
            for resource, value in baseline_resources.items()
        }

    def _estimate_performance_impact(self, scale_factor: float) -> Dict[str, Any]:
        """Estimate performance impact of scaling."""
        baseline_performance = self._calculate_baseline_performance()

        # Performance typically improves sub-linearly with scale
        performance_factor = scale_factor ** 0.8

        return {
            "projected_pages_per_hour": baseline_performance.get('pages_per_hour', 0) * performance_factor,
            "projected_response_time": baseline_performance.get('average_response_time', 0) / performance_factor,
            "performance_improvement_percent": (performance_factor - 1) * 100
        }

    def _calculate_monthly_costs(self) -> Dict[str, float]:
        """Calculate monthly costs."""
        now = datetime.now(timezone.utc)
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        current_month_costs = [
            c for c in self.crawl_costs
            if c.start_time >= current_month_start
        ]

        current_month_cost = sum(
            self.calculate_job_cost(c.job_id).get('total_cost', 0)
            for c in current_month_costs
        )

        return {
            "current_month": current_month_cost
        }

    def _project_monthly_cost(self) -> float:
        """Project monthly cost based on current trends."""
        monthly_costs = self._calculate_monthly_costs()
        current_cost = monthly_costs.get('current_month', 0)

        # Simple projection based on days elapsed
        now = datetime.now(timezone.utc)
        days_elapsed = now.day

        if days_elapsed == 0:
            return current_cost

        return (current_cost / days_elapsed) * 30

    def _calculate_current_month_spending(self) -> float:
        """Calculate current month spending."""
        return self._calculate_monthly_costs().get('current_month', 0)

    def _determine_budget_status(self, utilization: float) -> str:
        """Determine budget status based on utilization."""
        if utilization < 50:
            return "healthy"
        elif utilization < 80:
            return "moderate"
        elif utilization < 95:
            return "warning"
        else:
            return "critical"

    def _calculate_daily_costs(self) -> Dict[str, float]:
        """Calculate daily costs for historical analysis."""
        daily_costs = defaultdict(float)

        for crawl_cost in self.crawl_costs:
            day_key = crawl_cost.start_time.date().isoformat()
            job_cost = self.calculate_job_cost(crawl_cost.job_id)
            if "error" not in job_cost:
                daily_costs[day_key] += job_cost['total_cost']

        return dict(daily_costs)

    def _calculate_cost_trend(self, daily_costs: Dict[str, float]) -> float:
        """Calculate cost trend slope."""
        if len(daily_costs) < 3:
            return 0.0

        # Simple linear regression
        days = list(range(len(daily_costs)))
        costs = list(daily_costs.values())

        n = len(days)
        sum_x = sum(days)
        sum_y = sum(costs)
        sum_xy = sum(days[i] * costs[i] for i in range(n))
        sum_x2 = sum(x * x for x in days)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        return slope

    def _calculate_average_job_cost(self, crawl_costs: List[CrawlCost]) -> float:
        """Calculate average job cost for a list of crawl costs."""
        if not crawl_costs:
            return 0.0

        total_cost = 0
        valid_jobs = 0

        for crawl_cost in crawl_costs:
            job_cost = self.calculate_job_cost(crawl_cost.job_id)
            if "error" not in job_cost:
                total_cost += job_cost['total_cost']
                valid_jobs += 1

        return total_cost / valid_jobs if valid_jobs > 0 else 0.0

    def _check_real_time_budget_alerts(self, crawl_cost: CrawlCost):
        """Check for real-time budget alerts."""
        if not self.monthly_budget:
            return

        current_spending = self._calculate_current_month_spending()
        utilization = (current_spending / self.monthly_budget) * 100

        # Check monthly threshold
        if utilization > self.alert_thresholds['monthly_budget_percent']:
            alert = BudgetAlert(
                alert_id=f"monthly_budget_{int(time.time())}",
                alert_type="monthly_budget_threshold",
                severity=AlertSeverity.HIGH,
                title="Monthly Budget Threshold Exceeded",
                message=f"Monthly spending has reached {utilization:.1f}% of budget",
                current_value=utilization,
                threshold_value=self.alert_thresholds['monthly_budget_percent']
            )
            self.budget_alerts.append(alert)

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if still valid."""
        if cache_key in self._cost_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (datetime.now(timezone.utc) - cache_time).seconds < self._cache_ttl:
                return self._cost_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Any):
        """Cache a result with timestamp."""
        self._cost_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now(timezone.utc)

    def _invalidate_cache(self, cache_keys: List[str]):
        """Invalidate specific cache entries."""
        for key in cache_keys:
            if key in self._cost_cache:
                del self._cost_cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]

    async def _cleanup_old_data(self):
        """Clean up old data based on retention policies."""
        # Clean up old crawl costs (keep last 90 days)
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=90)
        self.crawl_costs = [c for c in self.crawl_costs if c.start_time >= cutoff_time]

        # Clean up old resource usage (keep last 30 days)
        resource_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        self.resource_usage_history = [r for r in self.resource_usage_history if r.timestamp >= resource_cutoff]

        # Clean up old alerts (keep last 7 days)
        alert_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        self.budget_alerts = [a for a in self.budget_alerts if a.created_at >= alert_cutoff]


# Global cost optimization analytics instance
_cost_analytics_instance: Optional[CostOptimizationAnalytics] = None


async def get_cost_optimization_analytics(monthly_budget: Optional[float] = None) -> CostOptimizationAnalytics:
    """Get or create global cost optimization analytics instance."""
    global _cost_analytics_instance

    if _cost_analytics_instance is None:
        _cost_analytics_instance = CostOptimizationAnalytics(monthly_budget=monthly_budget)
        await _cost_analytics_instance.initialize()

    return _cost_analytics_instance


async def shutdown_cost_optimization_analytics():
    """Shutdown global cost optimization analytics instance."""
    global _cost_analytics_instance

    if _cost_analytics_instance is not None:
        await _cost_analytics_instance.shutdown()
        _cost_analytics_instance = None