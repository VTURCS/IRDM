"""
Synthetic Virginia Tech opportunity listings for retrieval-evaluation experiments.

Five listings covering distinct domains and role types:
  VT-UAV-01  undergraduate research (lab / GNC)
  VT-IOT-02  undergraduate research + hardware build
  VT-LNX-03  lab operations / infrastructure position
  VT-TA-04   teaching assistant role
  VT-WEB-05  software project (student-facing tooling)

Schema per listing:
    opp_id      -> str        unique id
    title       -> str
    department  -> str        owning department / unit
    description -> str        scope of the work, written like a real posting
    key_terms   -> list[str]  terms that an EXACT keyword match will exploit

Why `key_terms` matters for the study: the lexical retriever can only win when a
student's vocabulary literally overlaps these strings. The semantic retriever is
expected to win precisely when it does not. Keep `key_terms` curated and domain
specific - if you leak generic words like "Python" or "Docker" into this list,
you blur the exact-vs-semantic contrast the evaluation is built to measure.

These are SYNTHETIC. They are written to resemble real VT postings so the
harness is executable before the real corpus is scraped.
"""

from __future__ import annotations

from typing import Dict, List

OPPORTUNITIES: List[Dict[str, object]] = [
    {
        "opp_id": "VT-UAV-01",
        "title": "UAV Trajectory Optimization for Aerial Surveying",
        "department": "Aerospace and Ocean Engineering",
        "description": (
            "Develop and validate trajectory optimization methods for small "
            "uncrewed aerial vehicles performing terrain survey missions. The "
            "student will formulate waypoint and continuous-time optimal control "
            "problems, implement them in MATLAB, and verify resulting flight "
            "paths in a Gazebo simulation before flight-testing on the lab's "
            "quadrotor platform. Emphasis on energy-optimal routing under wind "
            "disturbance and on producing reproducible simulation results."
        ),
        "key_terms": [
            "UAV",
            "trajectory optimization",
            "optimal control",
            "waypoint planning",
            "quadrotor",
            "aerial survey",
            "MATLAB",
            "Gazebo",
            "flight path",
            "guidance and control",
        ],
    },
    {
        "opp_id": "VT-IOT-02",
        "title": "Embedded IoT Sensor Nodes for Precision Irrigation",
        "department": "Electrical and Computer Engineering",
        "description": (
            "Design battery-powered embedded sensor nodes that measure soil "
            "moisture and transmit readings from agricultural test plots. Work "
            "includes writing embedded C firmware for ARM Cortex-M microcontrollers, "
            "bringing up I2C and SPI sensor drivers under an RTOS, laying out a "
            "low-power PCB, and tuning duty cycles so nodes survive a full growing "
            "season on primary cells. Field deployment alongside Biological Systems "
            "Engineering collaborators."
        ),
        "key_terms": [
            "embedded C",
            "firmware",
            "ARM Cortex-M",
            "microcontroller",
            "RTOS",
            "I2C",
            "SPI",
            "sensor driver",
            "PCB design",
            "low-power",
            "soil moisture",
            "LoRaWAN",
        ],
    },
    {
        "opp_id": "VT-LNX-03",
        "title": "Research Computing Infrastructure and Linux Systems Administration",
        "department": "Computer Science",
        "description": (
            "Maintain the systems that support a research group's experiments: "
            "Debian servers, a Proxmox virtualization cluster, ZFS storage with "
            "snapshot replication, and containerized workloads. The student will "
            "write Ansible roles to make configuration reproducible, author systemd "
            "units for lab services, manage VLANs and firewall rules, and build "
            "Prometheus and Grafana dashboards so failures are caught before "
            "deadlines are missed. Includes a documented backup and restore drill."
        ),
        "key_terms": [
            "Linux",
            "systems administration",
            "Debian",
            "Bash",
            "Ansible",
            "systemd",
            "Proxmox",
            "ZFS",
            "Docker",
            "networking",
            "VLAN",
            "Prometheus",
            "backup",
            "uptime",
        ],
    },
    {
        "opp_id": "VT-TA-04",
        "title": "Undergraduate Teaching Assistant - First-Year Engineering (ENGE 1215)",
        "department": "Engineering Education",
        "description": (
            "Support instruction in the first-year engineering design sequence. "
            "Duties include leading MATLAB lab sections for roughly 25 students, "
            "holding weekly office hours, grading design notebooks against a "
            "rubric, and coaching student teams through open-ended design projects. "
            "Successful applicants are patient, communicate clearly to students "
            "with no prior programming background, and can give feedback that "
            "improves work without discouraging beginners."
        ),
        "key_terms": [
            "teaching assistant",
            "first-year engineering",
            "ENGE 1215",
            "MATLAB",
            "grading",
            "rubric",
            "office hours",
            "mentoring",
            "design notebooks",
            "instruction",
            "communication",
        ],
    },
    {
        "opp_id": "VT-WEB-05",
        "title": "Full-Stack Developer - Campus Sustainability Dashboard",
        "department": "Computer Science",
        "description": (
            "Build the public web application that reports campus energy and "
            "emissions data. The stack is React with TypeScript on the front end "
            "and Node.js with PostgreSQL behind a REST API. The student will "
            "implement interactive data visualizations, wire up Docker Compose and "
            "GitHub Actions for continuous integration and deployment, and meet "
            "WCAG accessibility requirements so the dashboard is usable with a "
            "screen reader."
        ),
        "key_terms": [
            "React",
            "TypeScript",
            "Node.js",
            "PostgreSQL",
            "REST API",
            "full-stack",
            "Docker",
            "CI/CD",
            "GitHub Actions",
            "accessibility",
            "WCAG",
            "data visualization",
            "responsive design",
        ],
    },
]
