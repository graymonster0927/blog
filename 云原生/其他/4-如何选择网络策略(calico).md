## Kubernetes网络基础模型

Kubernetes 网络模型定义了一个“扁平”的网络，在这个网络中：

* 每个 Pod 都有自己的 IP 地址。
* 任何节点上的 Pod 都可以直接与所有其他节点上的 Pod 通信，无需进行网络地址转换（NAT）。

* 这样就可以创建一个干净、向后兼容的模型，使得 Pod 可以从端口分配、命名、服务发现、负载均衡、应用程序配置和迁移等方面像虚拟机或物理主机一样进行处理。网络分割可以使用网络策略来定义，以限制这些基本网络功能内的流量。

在这个模型中，支持不同的网络方法和环境有很大的灵活性。网络的实现细节取决于使用的 CNI、网络和云提供商插件的组合。

## CNI
CNI（容器网络接口）是一个标准 API，允许不同的网络实现插入到 Kubernetes 中。每当创建或销毁一个 Pod 时，Kubernetes 就会调用该 API。CNI 插件分为两种类型：

* CNI 网络插件：负责将 Pod 添加到或从 Kubernetes Pod 网络中删除。这包括创建/删除每个 Pod 的网络接口，并将其连接/断开到网络实现的其余部分。
* CNI IPAM 插件：负责为创建或删除的 Pod 分配和释放 IP 地址。根据插件的不同，这可能包括为每个节点分配一个或多个 IP 地址范围（CIDR），或从底层公共云的网络中获取 IP 地址来分配给 Pod。

## Kubenet
Kubenet 是 Kubernetes 中内置的一个极其基础的网络插件。它不实现跨节点网络或网络策略。通常，它与云提供商集成一起使用，为节点之间的通信在云提供商网络中设置路由，或在单节点环境中使用。Kubenet 不兼容 Calico。

## Overlay Network
覆盖网络是一种位于另一个网络之上的网络。在 Kubernetes 的上下文中，覆盖网络可以用于处理节点之间的 Pod 之间的流量，该流量位于不知道 Pod IP 地址或哪些 Pod 在哪些节点上运行的底层网络之上。覆盖网络通过将底层网络不知道如何处理的网络数据包（例如使用 Pod IP 地址）封装在外部数据包中来工作（例如节点 IP 地址），底层网络知道如何处理外部数据包。用于封装的两个常见网络协议是 VXLAN 和 IP-in-IP。

使用覆盖网络的主要优点是它减少了对底层网络的依赖性。例如，您可以在几乎任何底层网络之上运行 VXLAN 覆盖层，而无需与底层网络集成或进行任何更改。

使用覆盖网络的主要缺点是：

会稍微影响性能。封装数据包的过程需要一定的 CPU，并且编码封装所需的额外字节（VXLAN 或 IP-in-IP 标头）会减小可以发送的内部数据包的最大大小，这反过来可能意味着需要发送更多的数据包来发送相同数量的总数据。
Pod IP 地址在集群外无法路由。下面会有更多说明！

## Cross-subnet overlays
除了标准的 VXLAN 或 IP-in-IP overlay 外，Calico 还支持 VXLAN 和 IP-in-IP 的“跨子网”模式。在此模式下，每个子网内的底层网络充当 L2 网络。单个子网内发送的数据包不会进行封装，因此您可以获得非叠加网络的性能。跨子网发送的数据包会像普通叠加网络一样进行封装，减少了对底层网络的依赖（无需与底层网络集成或进行任何更改）。

与标准叠加网络一样，底层网络不知道 pod IP 地址，而且 pod IP 地址在集群外不可路由。

## 集群外 pod ip 的路由能力
不同Kubernetes网络实现的一个重要区别特征是pod IP地址是否可以在更广泛的网络中(集群之外)路由。

### 不可路由
不可路由的情况下，当一个 Pod 尝试与集群外的 IP 地址建立网络连接时，Kubernetes 会使用一种叫做 SNAT（源网络地址转换）的技术，将源 IP 地址从 Pod 的 IP 地址改为托管该 Pod 的节点的 IP 地址。连接上的任何返回数据包都会自动映射回该 Pod 的 IP 地址。因此，Pod 并不知道 SNAT 的发生，连接的目的地会将节点视为连接的源头，而底层更广泛的网络则从未看到 Pod 的 IP 地址。
对于相反方向的连接，即集群外部的某个对象需要连接到 Pod 时，这只能通过 Kubernetes 服务或 Kubernetes 入口实现。集群外部的任何内容都不能直接连接到 Pod IP 地址，因为更广泛的网络不知道如何将数据包路由到 Pod IP 地址。

### 可路由

如果 Pod IP 地址在集群外可路由，则 Pod 可以在不进行 SNAT 的情况下连接到外部世界，并且外部世界可以直接连接到 Pod，而无需通过 Kubernetes 服务或 Kubernetes 入口。

Pod IP 地址在集群外可路由的优点是：

避免对于外部连接的 SNAT 可能对于与现有更广泛的安全要求集成非常重要。它也可以简化调试和操作日志的可理解性。
如果您有专门的工作负载，意味着某些 Pod 需要直接访问而不经过 Kubernetes 服务或 Kubernetes 入口，则可路由的 Pod IP 比使用主机网络化的 Pod 更为操作简单。
Pod IP 地址在集群外可路由的主要缺点是，Pod IP 必须在更广泛的网络上是唯一的。因此，例如，如果运行多个集群，则需要为每个集群中的 Pod 使用不同的 IP 地址范围（CIDR）。这反过来可能会在规模运行或如果存在其他重要的企业需求时，导致 IP 地址范围耗尽的挑战。

### 决定路由性的因素是什么？

如果您的集群使用覆盖网络，则 Pod IP 通常不可路由到集群外。

如果您没有使用覆盖网络，则 Pod IP 地址是否在集群外可路由取决于使用的 CNI 插件、云提供商集成或（对于本地）与物理网络进行 BGP 对等连接的组合。

## BGP
BGP（边界网关协议）是一种标准化的网络协议，用于在网络中共享路由。它是互联网的基本构建块之一，具有出色的可扩展性特征。

Calico内置了对BGP的支持。在本地部署中，这使得Calico可以与物理网络进行对等连接（通常是到顶部路由器），以交换路由，从而创建一个非覆盖网络，使得Pod IP地址可以跨越更广泛的网络路由，就像任何其他连接到网络的工作负载一样。

## Calico

Calico的灵活模块化网络架构包括以下内容。

### Calico CNI网络插件

Calico CNI网络插件使用一对虚拟以太网设备（veth对）将Pod连接到主机网络命名空间的L3路由。这种L3架构避免了许多其他Kubernetes网络解决方案中存在的不必要复杂性和性能开销，如额外的L2桥接。

### Calico CNI IPAM插件

Calico CNI IPAM插件根据一个或多个可配置的IP地址范围为Pod分配IP地址，根据需要动态分配每个节点的小块IP地址。结果是与许多其他CNI IPAM插件（包括用于许多网络解决方案的主机本地IPAM插件）相比，更有效地使用IP地址空间。

### Overlay network modes

Calico可以提供VXLAN或IP-in-IP覆盖网络，包括仅跨子网的模式。

### Non-overlay network modes

Calico可以在任何底层L2网络或L3网络上提供非覆盖网络，或者是具有适当的云提供商集成的公共云网络或具有BGP功能的网络（通常是具有标准顶部交换机路由器的本地网络）。

### 网络策略执行

Calico的网络策略执行引擎实现了完整的Kubernetes网络策略功能，以及Calico网络策略的扩展功能。这与Calico的内置网络模式或任何其他Calico兼容的网络插件和云提供商集成配合使用。

除了Calico CNI插件和内置网络模式外，Calico还兼容许多第三方CNI插件和云提供商集成。

#### Amazon VPC CNI

Amazon VPC CNI插件从底层的AWS VPC中分配pod IP，并使用AWS弹性网络接口提供VPC本地pod网络（可以在集群外部路由的pod IP）。它是Amazon EKS中默认使用的网络，配合Calico用于网络策略执行。

#### Azure CNI

Azure CNI插件从底层的Azure VNET中分配pod IP，并配置Azure虚拟网络以提供VNET本地pod网络（可以在集群外部路由的pod IP）。它是Microsoft AKS中默认使用的网络，配合Calico用于网络策略执行。

#### Azure云提供商

Azure云提供商集成可用作Azure CNI插件的替代方案。它使用host-local IPAM CNI插件分配pod IP，并在底层的Azure VNET子网中配置相应的路由。Pod IP只能在VNET子网内路由（通常意味着它们不能在集群外部路由）。

#### Google云提供商

Google云提供商集成使用host-local IPAM CNI插件分配pod IP，并在Google云网络别名IP范围中编程以在Google云上提供VPC本地pod网络（可以在集群外部路由的pod IP）。它是Google Kubernetes Engine（GKE）的默认网络，配合Calico用于网络策略执行。

#### 主机本地IPAM

主机本地CNI IPAM插件是一种常用的IP地址管理CNI插件，它为每个节点分配一个固定大小的IP地址范围（CIDR），然后从该范围内分配pod IP地址。默认地址范围大小为256个IP地址（/24），但其中两个IP地址保留用于特殊用途，不分配给pod。主机本地CNI IPAM插件的简单性使其易于理解，但与Calico CNI IPAM插件相比，导致IP地址空间使用效率较低。

#### Flannel

Flannel使用从host-local IPAM CNI插件获得的每个节点的静态CIDR路由pod流量。Flannel提供许多网络后端，但主要与其VXLAN覆盖后端一起使用。Calico CNI和Calico网络策略可以与flannel和host-local IPAM插件组合使用，以提供带有策略执行的VXLAN网络。这种组合有时被称为“Canal”。

注意：Calico现在已经内置支持VXLAN，我们通常建议优先使用它以简化使用，而不是使用Calico+Flannel的组合。


## 原文
https://docs.tigera.io/calico/latest/networking/determine-best-networking#big-picture
