int x,y,z;

if(i < NRX){
    x = rxcoords[INDEX2D_RXCOORDS(i,0)];
    y = rxcoords[INDEX2D_RXCOORDS(i,1)];
    z = rxcoords[INDEX2D_RXCOORDS(i,2)];
    rxs[INDEX3D_RXS(0,iteration,i)] = Ex[INDEX3D_FIELDS(x,y,z)];
    rxs[INDEX3D_RXS(1,iteration,i)] = Ey[INDEX3D_FIELDS(x,y,z)];
    rxs[INDEX3D_RXS(2,iteration,i)] = Ez[INDEX3D_FIELDS(x,y,z)];
    rxs[INDEX3D_RXS(3,iteration,i)] = Hx[INDEX3D_FIELDS(x,y,z)];
    rxs[INDEX3D_RXS(4,iteration,i)] = Hy[INDEX3D_FIELDS(x,y,z)];
    rxs[INDEX3D_RXS(5,iteration,i)] = Hz[INDEX3D_FIELDS(x,y,z)];
}